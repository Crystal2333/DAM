import torch
import torch.nn as nn
import gaze_model
import resnet_scene


class GazeTargetDetectionNet(nn.Module):
    def __init__(self):
        super(GazeTargetDetectionNet, self).__init__()
        self.conv_output = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv_inout = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.fc_inout = nn.Linear(49, 1)

        self.deconv_output = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, stride=1)
        )
        
        self.gazenet = gaze_model.GazeNet()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        
        self.scene_saliency = resnet_scene.resnet50(pretrained=True)
        self.scene_saliency.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)


    def forward(self, x):
        img, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field = x
        _, height, width = depthmap.size()

        # 3D gaze direction estimation
        gaze_input = head, left_eye, right_eye, eye_weight
        gaze_direction, _ = self.gazenet(gaze_input)
        z = torch.mul(torch.cos(gaze_direction[:,1]), torch.cos(gaze_direction[:,0]))
        z = z.view(-1, 1, 1)
        #print(z)
        x = torch.mul(torch.cos(gaze_direction[:,1]), torch.sin(gaze_direction[:,0]))
        x = torch.mul(x, -1).view(-1, 1)
        #print(x)
        y = torch.mul(torch.sin(gaze_direction[:,1]), -1).view(-1, 1)
        #print(y)
        direction = torch.cat((x, y), 1)
        norm = torch.norm(direction, 2, dim=1)
        normalized_direction = direction / norm.view(-1, 1)
        #print(normalized_direction)

        # generate gaze field map
        gaze_field = gaze_field.permute(0, 2, 3, 1).contiguous()
        gaze_field = gaze_field.view(-1, height*width, 2)
        gaze_field = torch.matmul(gaze_field, normalized_direction.view(-1, 2, 1))
        gaze_field_map = gaze_field.view(-1, height, width, 1)
        gaze_field_map = gaze_field_map.permute(0, 3, 1, 2).contiguous()
        gaze_field_map = self.relu(gaze_field_map)
        gaze_field_map = torch.pow(gaze_field_map, 5)

        # front/mid/back scene segmentation
        depth_value = torch.mul(depthmap, head_channel).view(-1, height*width)
        depth_value = torch.sum(depth_value, dim=1).view(-1, 1, 1)
        depth_map = torch.sub(depthmap, depth_value)

        front_map = self.relu(depth_map)
        back_map = self.relu(-depth_map)
        mid_map = torch.sub(1,torch.mul(16, torch.pow(depth_map, 2)))
        mid_map = self.relu(mid_map)

        depth_map = self.relu(z-0.3)*front_map + self.relu(-z-0.3)*back_map + self.relu(-z+0.3)*self.relu(z+0.3)*mid_map
        depth_map = depth_map.view(-1, 1, height, width)
        # attention map
        attention_map = torch.mul(depth_map, gaze_field_map)

        # output heatmap
        fused_feat = torch.cat((img, attention_map), 1)
        fused_feat = self.scene_saliency(fused_feat)
        output = self.conv_output(fused_feat)
        output = self.deconv_output(output)

        # inout cls_head
        inout = self.conv_inout(fused_feat)
        inout = inout.view(-1, 49)
        inout = self.fc_inout(inout)

        return normalized_direction, output, inout


if __name__ == '__main__':
    img = torch.rand((4, 3, 224, 224))
    depthmap = torch.rand(4, 224, 224)
    head = torch.rand((4, 3, 224, 224))
    head_channel = torch.zeros(4, 224, 224)
    head_channel[:, 1, 1] = 1
    left_eye = torch.rand((4, 3, 36, 60))
    right_eye = torch.rand((4, 3, 36, 60))
    eye_weight = torch.zeros((4, 1))
    gaze_field = torch.rand((4, 2, 224, 224))

    x = img, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field
    model = GazeTargetDetectionNet()
    normalized_direction, output, inout = model(x)
    print(normalized_direction.shape, output.shape, inout.shape)
