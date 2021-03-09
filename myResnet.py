#参考:https://github.com/HaoMood/blinear-cnn-faster/blob/master/src/model.py

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F


class resnet_18_pre(nn.Module):
    def __init__(self, num_labels=12):
        super(resnet_18_pre, self).__init__()
        self.feature = torchvision.models.resnet18(pretrained=False)
        self.feature = torch.nn.Sequential(*list(self.feature.children())[:-1])
        self.fc = nn.Linear(in_features=512, out_features=num_labels, bias=True)

    def forward(self, x):
        x = self.feature(x)
        # the shape before view is  torch.Size([128, 512, 1, 1])
        x = x.view(x.shape[0], -1)
        # the shape after view is  torch.Size([128, 512])
        out = self.fc(x)
        return out



class resnet_101_pre(nn.Module):
    def __init__(self, num_labels=12):
        super(resnet_101_pre, self).__init__()
        self.feature = torchvision.models.resnet101(pretrained=True)
        #只提到[:-4]图片是(512, 28, 28), [:-3]是(1024, 14, 14)
        print(self.feature)
        self.feature = torch.nn.Sequential(*list(self.feature.children())[:-3])
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) #TODO:老师提到AvgPool不一定好，没有了空间信息
        self.fc = nn.Linear(in_features=1024, out_features=num_labels, bias=True)
        

    def forward(self, x):
        x = self.feature(x) # (64, 1024, 14, 14) 
        x = self.AvgPool(x)# (64, 1024, 1, 1)
        x = x.view(x.shape[0], -1)  #(64, 1024)
        # print(x.shape)
        out = self.fc(x)    # (64, 14)
        return out


class MobileNet_pre(nn.Module):
    def __init__(self, num_labels=12):
        super(MobileNet_pre, self).__init__()
        self.feature = torchvision.models.mobilenet_v2()
        self.fc = nn.Linear(in_features=1000, out_features=num_labels, bias=True)
    

    def forward(self, x):
        x = self.feature(x)
        out = self.fc(x)
        return out

class mobilenet_test(nn.Module):

    def __init__(self, num_labels=12):
        super(mobilenet_test, self).__init__()
        self.feature = torchvision.models.mobilenet_v2()
        self.feature = torch.nn.Sequential(*list(self.feature.children())[:-1])
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.drop_fc = nn.Sequential(OrderedDict([
            ('0', nn.Dropout(p=0.4, inplace=False)),
            ('1', nn.Linear(in_features=1280, out_features=num_labels, bias=True))
        ]))

    def forward(self, x):
        x = self.feature(x)
        x = self.AvgPool(x)# (64, 1024, 1, 1)
        x = x.view(x.shape[0], -1)  #(64, 1024)
        out = self.drop_fc(x)
        return out



#自己搭一个resnet34来试一试先

class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):  # 224x224x3
    # 实现主module:ResNet34
    def __init__(self, num_classes=1):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # (224+2*p-)/2(向下取整)+1，size减半->112
            nn.BatchNorm2d(64),  # 112x112x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64

        # 重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理，但是为了统一操作。。。
        self.layer2 = self.make_layer(64, 128, 4, stride=2)  # 第一个stride=2,剩下3个stride=1;28x28x128
        self.layer3 = self.make_layer(128, 256, 6, stride=2)  # 14x14x256
        self.layer4 = self.make_layer(256, 512, 3, stride=2)  # 7x7x512
        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        x = F.avg_pool2d(x, 7)  # 1x1x512
        x = x.view(x.size(0), -1)  # 将输出拉伸为一行：1x512
        x = self.fc(x)  # 1x1
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间


