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
    def __init__(self):
        super().__init__()
        self.register_buffer('rws', torch.arange(0, 256).float().unsqueeze(1))
        self.register_buffer('cls', torch.arange(0, 256).float().unsqueeze(0))


    def forward(self, input):
        l = F.softmax(input.view((input.shape[0], input.shape[1], -1)), dim=2).view(input.shape)
        xs = ((l * self.state_dict()['rws']).sum(dim=(2, 3))).unsqueeze(2)
        ys = ((l * self.state_dict()['cls']).sum(dim=(2, 3))).unsqueeze(2)
        l = torch.cat((xs, ys), dim=2).view(input.shape[0], -1)
        return l


class SoftResNetModel(nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        self.avgpool = nn.ConvTranspose2d(in_channels=256, out_channels=17, stride=17, kernel_size=3, padding=1)
        self.argm = SoftArgmax()

        if pretrained:
            self.res_conv.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, inputs, filename):
        x, _ = self.res_conv(inputs)
        x = self.avgpool(x)
        x = self.argm(x)
        return x


class TransUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding):
        super().__init__()
        self.ups = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                      kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        # self.rl = nn.ReLU()

    def forward(self, input, skip=None):
        if skip is not None:
            input = torch.cat((skip, input), dim=1)
        x = self.ups(input)
        x = self.bn(x)
        # x = self.rl(x)
        return x

class SingleUpsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        self.l1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.l2 = nn.Upsample((256, 256))
        self.sgn = nn.Sigmoid()

    def forward(self, inputs, filename=''):
        x, _ = self.res_conv(inputs)
        x = self.l1(x)
        x = self.l2(x)
        x = self.sgn(x)
        return x.squeeze(1)


class TripleUpsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        self.l1 = TransUpsampling(in_channels=256, out_channels=128, stride=2, kernel_size=2, padding=0)
        self.l2 = TransUpsampling(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1)
        self.l3 = TransUpsampling(in_channels=64, out_channels=32, stride=2, kernel_size=6, padding=2)
        self.l4 = TransUpsampling(in_channels=32, out_channels=1, stride=2, kernel_size=8, padding=3)
        self.sgn = nn.Sigmoid()

    def forward(self, inputs, filename=''):
        x, _ = self.res_conv(inputs)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.sgn(x)
        return x.squeeze(1)

class TripleUpsamplingSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_conv = ResNetConv(BasicBlock, [2, 2, 2, 2])
        self.l1 = TransUpsampling(in_channels=512, out_channels=128, stride=2, kernel_size=2, padding=0)
        self.l2 = TransUpsampling(in_channels=256, out_channels=64, stride=2, kernel_size=4, padding=1)
        self.l3 = TransUpsampling(in_channels=128, out_channels=32, stride=2, kernel_size=6, padding=2)
        self.l4 = TransUpsampling(in_channels=32, out_channels=1, stride=2, kernel_size=8, padding=3)
        self.sgn = nn.Sigmoid()

    def forward(self, inputs, filename=''):
        x, skips = self.res_conv(inputs)
        x = self.l1(x, skips[2])
        x = self.l2(x, skips[1])
        x = self.l3(x, skips[0])
        x = self.l4(x)
        x = self.sgn(x)
        return x.squeeze(1)


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
