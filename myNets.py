import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#resnet34 stem network
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
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # (224+2*p-)/2(向下取整)+1，size减半->112
            nn.BatchNorm2d(64),  # 112x112x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64

        # 重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64,layer1层输入输出一样
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
        x = self.layer4(x)  # 7x7x51
        x = F.avg_pool2d(x, 7)  # 1x1x512
        x = x.view(x.size(0), -1)  # 将输出拉伸为一行：1x512
        x = self.fc(x)  # 1x1     这里也截取一下  原先resnet34得部分
        return x  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12得

class Lnet(ResNet34):
    def __init__(self, num_classes):
        super(Lnet,self).__init__(num_classes)
        self.lstm_list = []
        for i in range(num_classes):
            self.lstm = nn.LSTM(512, 49, 1, dropout=0.2).to('cuda:1')
            self.lstm_list.append(self.lstm)

    def init(self,x):
        h0 = torch.empty(1, x.shape[0], 49).to('cuda:1')  #初始化参数（非必要）
        c0 = torch.empty(1, x.shape[0], 49).to('cuda:1')
        nn.init.xavier_normal_(h0,gain=20)   #参数初始化
        nn.init.xavier_normal_(c0,gain=20)

    def reshape_data(self,x):
        x = x.reshape([x.shape[0], 512, 49])  # 128x512x49
        x = x.transpose(0, 2)  # 49x512x128
        input = x.transpose(1, 2)  # 49x128x512
        return input

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        self.init(x)
        input=self.reshape_data(x)
        out=torch.zeros([input.shape[1],12]).to('cuda:1')   #LSTM
        for i in range(12):
            output,(hn,cn)=self.lstm_list[i](input)
            hn=hn.mean(2)#128x1
            out[:,i]=torch.add(out[:,i],hn)
        return out  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12得

class Transformer(ResNet34):
    def __init__(self, num_classes):
        super(Transformer,self).__init__(num_classes)
        # #Encoder 分成12个Encoder
        # self.transformer_list = []
        # for i in range(num_classes):
        #     self.transformer = TransformerModel(ninp=512, nhead=4, nhid=512, nlayers=2, dropout=0.5).to('cuda:1')
        #     self.transformer_list.append(self.transformer)
        #只用一个
        self.transformer = TransformerModel(num=num_classes,ninp=512, nhead=4, nhid=512, nlayers=2, dropout=0.5).to('cuda:1')

    def reshape_data(self,x):
        x = x.reshape([x.shape[0], 512, 49])  # 128x512x49
        x = x.transpose(0, 2)  # 49x512x128
        input = x.transpose(1, 2)  # 49x128x512
        return input

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        input = self.reshape_data(x)
        # Encoder 12个
        # out=torch.zeros([input.shape[0],12]).to('cuda:1')
        # for i in range(12):
        #     output=self.transformer_list[i](input) #200x49x49  bs x 49 x 512
        #     output=output[:,:,-1].view(output.size(0),-1)
        #     output=output.mean(1)
        #     out[:,i]=torch.add(out[:,i],output)
        # 一个Encoder
        out = self.transformer(input)
        return out  # 1x1，将结果化为(0~1)之间 最后得输出肯定是128x12得

#Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, num, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, num)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output=output.mean(0)
        output = self.decoder(output)
        return output

