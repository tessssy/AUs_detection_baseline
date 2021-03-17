#参考:https://github.com/HaoMood/blinear-cnn-faster/blob/master/src/model.py

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F
#resnet34

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
    def __init__(self, num_classes=12):
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
        self.bn = nn.BatchNorm1d(49).to('cuda:1')

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
        #取到7x7的每一个像素送进lstm
        # h0 = torch.empty(49, x.shape[0], 1).to('cuda:1')
        # c0 = torch.empty(49, x.shape[0], 1).to('cuda:1')
        x=x.reshape([x.shape[0],512,49])

        x=x.transpose(0,2)
        input=x.transpose(1,2) #49x128x512
        # rnn = nn.LSTM(512, 512, 2)
        # rnn.to('cuda:0')
        # output,(hn,cn) = rnn(input)
        # output = output.mean(0)
        # output = self.fc(output)
        # nn.init.xavier_normal_(h0,gain=20)
        # nn.init.xavier_normal_(c0,gain=20)
        out=torch.zeros([input.shape[1],12]).to('cuda:1')
        for i in range(12):
            rnn = nn.LSTM(512, 49, 1,dropout=0.2)
            rnn.to('cuda:1')
            output,(hn,cn)=rnn(input)
            hn=hn.transpose(1,2)
            hn=self.bn(hn)
#            hn=hn.mean(1)#128x1
            out[:,i]=torch.add(out[:,i],hn[:,-1,:])
        # x = F.avg_pool2d(x, 7)  # 1x1x512
        # x = x.view(x.size(0), -1)  # 将输出拉伸为一行：1x512
        # x = self.fc(x)  # 1x1     这里也截取一下
        return out  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12得


