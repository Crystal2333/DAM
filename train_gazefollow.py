import os
import tqdm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from dataset import Gazefollowing_Dataset
from model import GazeTargetDetectionNet
from util import loggging, auc, L2_dist, multi_hot_targets, argmax_pts

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
data_dir = '/data/gazefollow'
depth_dir = '/data/gazefollow/depthmap'
train_annotation_dir = '/data/gazefollow/annotations/train_annotations_release.txt'
test_annotation_dir = '/data/gazefollow/annotations/test_annotations_release.txt'

cosine_similarity = nn.CosineSimilarity()
mse_loss = nn.MSELoss(reduction='none')
bce_loss = nn.BCELoss(reduction='none')


def dist_loss(predicted_heatmap, gaze_point):
    predicted_heatmap = predicted_heatmap.view(-1, 64*64)
    a = torch.argmax(predicted_heatmap, dim=1)
    b = torch.cat(((a%64).view(-1,1), (a//64).view(-1,1)), dim=1).float()/64.0
    distloss = mse_loss(b, gaze_point)
    distloss = torch.mean(distloss, dim=1)
    return distloss


def gaze_loss(predicted_direction, groundtruth_direction):
    angle_loss = 1 - cosine_similarity(predicted_direction, groundtruth_direction)
    return angle_loss


def test(model, test_loader):
    model.eval()
    AUC = []
    Dist = []
    MDist = []
    Ang = []
    MAng = []

    with torch.no_grad():
        for i, (image, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field, gaze_collect, gaze_average, seperate_gaze_direction, average_gaze_direction, imsize, head_position) in enumerate(test_loader):
            input = image.cuda(), depthmap.cuda(), head.cuda(), head_channel.cuda(), left_eye.cuda(), right_eye.cuda(), eye_weight.cuda(), gaze_field.cuda()
            _, predicted_heatmap, predicted_inout = model(input)
            predicted_heatmap = predicted_heatmap.squeeze(1)
            #AUC
            multi_hot = multi_hot_targets(gaze_collect[0], imsize[0])
            pred_heatmap = Image.fromarray(predicted_heatmap[0].cpu().numpy())
            scaled_heatmap = pred_heatmap.resize((imsize[0][0], imsize[0][1]), resample=Image.BILINEAR)
            scaled_heatmap = np.array(scaled_heatmap)
            auc_score = auc(scaled_heatmap, multi_hot)
            AUC.append(auc_score)
            #Dist
            pred_x, pred_y = argmax_pts(predicted_heatmap[0].cpu().numpy())
            norm_p = [pred_x/64.0, pred_y/64.0]
            avg_dist = L2_dist(gaze_average[0].numpy(), norm_p)
            Dist.append(avg_dist)
            #MDist
            seperate_dist = []
            for gt_gaze in gaze_collect[0]:
                seperate_dist.append(L2_dist(gt_gaze.numpy(), norm_p))
            MDist.append(min(seperate_dist))
            #Ang
            predicted_direction = torch.Tensor([norm_p]) - head_position
            average_cos = cosine_similarity(predicted_direction.cpu(), average_gaze_direction).numpy()
            average_cos = np.maximum(np.minimum(average_cos, 1.0), -1.0)
            average_Ang = np.arccos(average_cos) * 180 / np.pi
            Ang.append(average_Ang)
            #MAng
            seperate_gaze_direction = seperate_gaze_direction.squeeze(0)
            seperate_cos = cosine_similarity(predicted_direction.cpu(), seperate_gaze_direction).numpy()
            seperate_cos = np.maximum(np.minimum(seperate_cos, 1.0), -1.0)
            seperate_Ang = np.arccos(seperate_cos) * 180 / np.pi
            MAng.append(np.min(seperate_Ang))

        test_AUC = np.mean(AUC)
        test_Dist = np.mean(Dist)
        test_MDist = np.mean(MDist)
        test_Ang = np.mean(Ang)
        test_MAng = np.mean(MAng)
    
    return test_AUC, test_Dist, test_MDist, test_Ang, test_MAng


def train(args, logger):
    model = GazeTargetDetectionNet()
    model = nn.DataParallel(model)
    '''
    model_dict = model.state_dict()
    pretrained_dict = torch.load('model.pth')
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    '''
    model.cuda()

    train_dataset = Gazefollowing_Dataset(data_dir, train_annotation_dir, depth_dir)
    test_dataset = Gazefollowing_Dataset(data_dir, test_annotation_dir, depth_dir, test=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['number_batch'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    gaze_params = list(map(id, model.module.gazenet.parameters()))
    sali_params = list(map(id, model.module.scene_saliency.parameters()))
    base_params = filter(lambda p: id(p) not in gaze_params + sali_params, model.module.parameters())
    
    optimizer = torch.optim.Adam([{'params': base_params},
                                  {'params': model.module.gazenet.parameters(), 'lr': args['learning_rate']},
                                  {'params': model.module.scene_saliency.parameters(), 'lr': args['learning_rate']}
                                 ], 
                                 lr=args['learning_rate'], weight_decay=args['weight_decay'])
    
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)

    print('Start training...')
    for i_epoch in tqdm.trange(args['number_epoch']):
        model.train()
        loss = []
        Angloss = []
        Dist = []
        logger.info('lr of epoch %s: %s' % (i_epoch+1, optimizer.param_groups[0]['lr']))
        for i, (image, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field, heatmap, gaze_direction, inout, imsize, gaze_point) in enumerate(train_loader):
            model.zero_grad()
            input = image.cuda(), depthmap.cuda(), head.cuda(), head_channel.cuda(), left_eye.cuda(), right_eye.cuda(), eye_weight.cuda(), gaze_field.cuda()
            heatmap = heatmap.cuda()
            inout = inout.cuda().squeeze(1)
            gaze_direction = gaze_direction.cuda()
            pred_direction, predicted_heatmap, predicted_inout = model(input)
            predicted_heatmap = predicted_heatmap.squeeze(1)

            l2_loss = mse_loss(predicted_heatmap, heatmap)*10000
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mul(l2_loss, inout) # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss)/torch.sum(inout)

            angle_loss = gaze_loss(pred_direction, gaze_direction)*10000
            angle_loss = torch.mul(angle_loss, inout)
            angle_loss = torch.sum(angle_loss)/torch.sum(inout)
            total_loss = l2_loss + angle_loss

            #Dist
            for idx in range(len(gaze_point)):
                pred_x, pred_y = argmax_pts(predicted_heatmap[idx].detach().cpu().numpy())
                norm_p = [pred_x/64.0, pred_y/64.0]
                avg_dist = L2_dist(gaze_point[idx].numpy(), norm_p)
                Dist.append(avg_dist)

            #print(total_loss)
            total_loss.backward()
            optimizer.step()

            loss.append(total_loss.data.cpu().numpy())
            Angloss.append(angle_loss.data.cpu().numpy())
            if i % 20 == 19:
                logger.info('epoch: %s, batch: %s, train_loss: %s, train_Dist: %s, Angloss: %s' % (i_epoch+1, i+1, np.mean(loss), np.mean(Dist), np.mean(Angloss)))
                loss.clear()
                Dist.clear()
                Angloss.clear()
        scheduler.step()

        if i_epoch % 1 == 0:
            test_AUC, test_Dist, test_MDist, test_Ang, test_MAng = test(model, test_loader)
            logger.info('--------------------testing--------------------')
            logger.info('epoch: %s, test_AUC: %s, test_Dist: %s, test_MDist: %s, test_Ang: %s, test_MAng: %s' % (i_epoch+1, test_AUC, test_Dist, test_MDist, test_Ang, test_MAng))


if __name__ == '__main__':
    args = {
        'learning_rate':0.0001,
        'number_batch':128,
        'number_epoch':20,
        'weight_decay':0.0001,
    }
    logger = loggging(filename='train_gazefollow.log')
    logger.info('learning_rate: %s, number_batch: %s, number_epoch: %s, weight_decay: %s' % (args['learning_rate'], args['number_batch'], args['number_epoch'], args['weight_decay']))
    train(args, logger)
