import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls


class ResNetConv(ResNet):    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        intermediate = []
        x = self.layer1(x); intermediate.append(x)
        x = self.layer2(x); intermediate.append(x)
        x = self.layer3(x); intermediate.append(x)
        
        return x, intermediate


class SoftArgmax(nn.Module):

    def forward(self, input):
        l = F.softmax(input.view((input.shape[0], input.shape[1], -1)), dim=2).view(input.shape)
        h = input.shape[2]
        w = input.shape[3]
        rws = torch.arange(0, h).float().unsqueeze(1)
        cls = torch.arange(0, w).float().unsqueeze(0)
        xs = ((l * rws).sum(dim=(2,3))).unsqueeze(2)
        ys = ((l * cls).sum(dim=(2,3))).unsqueeze(2)
        l = torch.cat((xs, ys), dim=2).view(input.shape[0], -1)
        return l


class SoftResNetModel(nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        self.avgpool = nn.AdaptiveAvgPool3d((17, 256, 256))
        self.argm = SoftArgmax()

        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, inputs, filename):
        x, _ = self.res_conv(inputs)
        x = self.avgpool(x)
        x = self.argm(x)
        return x


class ResNetModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        # base network
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        
        # other network modules
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 34)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, inputs, filename):
        x, _ = self.res_conv(inputs)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + 0.5
        return x
