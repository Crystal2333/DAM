import math
import torch
import torch.nn as nn
from torchvision import models


class GazeNet(nn.Module):
    def __init__(self, pretrained=False):
        super(GazeNet, self).__init__()
        self.pretrained = pretrained

        self.head_backbone = models.resnet18(pretrained=self.pretrained)
        self.head_backbone.fc = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256)
        )
        self.lstm = nn.LSTM(256, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.last_layer = nn.Linear(512, 3)

        self.left_eye_backbone = models.resnet18(pretrained=self.pretrained)
        self.left_eye_backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.right_eye_backbone = models.resnet18(pretrained=self.pretrained)
        self.right_eye_backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.eye_feature_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

        
    def forward(self, x):
        head, left_eye, right_eye, eye_weight = x

        head_out = self.head_backbone(head).view(-1, 1, 256)
        self.lstm.flatten_parameters()
        head_out, _ = self.lstm(head_out)
        head_out = head_out.view(-1, 512)
        head_out = self.last_layer(head_out)
        angular_out = head_out[:,:2]
        angular_out[:,0:1] = math.pi*nn.Tanh()(angular_out[:,0:1])
        angular_out[:,1:2] = (math.pi/2)*nn.Tanh()(angular_out[:,1:2])

        left_eye_feat = self.left_eye_backbone(left_eye)
        right_eye_feat = self.right_eye_backbone(right_eye)
        fused_eye_feat = torch.cat((left_eye_feat, right_eye_feat), 1)
        fused_eye_feat = self.eye_feature_fusion(fused_eye_feat)
        fused_eye_feat = fused_eye_feat * eye_weight
        
        fused_feat = torch.cat((fused_eye_feat, angular_out), 1)

        out = self.output_layer(fused_feat)

        ang_out = out[:,:2]
        ang_out[:,0:1] = math.pi*nn.Tanh()(ang_out[:,0:1])
        ang_out[:,1:2] = (math.pi/2)*nn.Tanh()(ang_out[:,1:2])

        var = math.pi*nn.Sigmoid()(out[:,2:3])
        var = var.view(-1,1).expand(var.size(0),2)
        return ang_out, var


class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)

        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90


if __name__ == '__main__':
    head = torch.rand((4, 3, 224, 224))
    left_eye = torch.rand((4, 3, 36, 60))
    right_eye = torch.rand((4, 3, 36, 60))
    eye_weight = torch.ones((4, 1))

    x = head, left_eye, right_eye, eye_weight
    model = GazeNet(pretrained=True)
    out, var = model(x)
    print(out.shape, var.shape)
