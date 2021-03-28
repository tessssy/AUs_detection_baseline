from torch.utils.data import  DataLoader
import torch.optim as optim
from tqdm import tqdm
from myNets import *
#from sklearn.metrics import f1_score
from MyDatasets import *
import torch.nn.functional as F
from torchvision import transforms
#import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

writer1=SummaryWriter('runs_18')

# ----------------------------------------------------------
# train on BP4D

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # net.module
    net.train()
    au_keys = ['au1', 'au2', 'au4', 'au6', 'au7', 'au10', 'au12', 'au14', 'au15', 'au17', 'au23', 'au24']
    train_loss = 0
    acc_in_epoch = [0 for i in range(len(au_keys))]
    total_in_epoch = [0 for i in range(len(au_keys))]
    train_acc_in_epoch = [0 for i in range(len(au_keys))]
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device).squeeze(dim=1).float()
        optimizer.zero_grad()
        # inputs.type/shape:torch.Tensor/[128,3,224,224]
        # targets的shape是batch_size * 12
        outputs = net(inputs)
        # functional.BCE_with_logits自范带对预测分数进行sigmoid操作，因此可以无所谓outputs的取值围
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean').to(device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        m = nn.Sigmoid()
        predicted = m(outputs)
        pred = [0 for i in range(len(au_keys))]
        for i in range(len(targets)):
            if(targets[i,:].nonzero().numel()!=0):
                index_1=targets[i,:].nonzero()
                for j in index_1:
                    pred[j]=pred[j]+predicted[i,j].item()
        total = []
        total_acc = []
        Sum = 0
        Not_nan = 0
        for i in range(len(au_keys)):
            total.append(targets[:, i].sum().item())
            if(total[i]!=0):
                acc = pred[i] / total[i]
                total_acc.append(acc)
            else:
                acc=0.5
                total_acc.append(acc)
            if (total_acc[i]!=0):
                Sum = total_acc[i] + Sum
                Not_nan = Not_nan + 1
        if (Not_nan == 0):
            print("This batch contains no AUs")
        else:
            # avg_acc = Sum / Not_nan                  #一个batch得平均准确率 和loss
            # print(str(batch_idx) + 'avg:{:.6f}|loss:{:.6f}'.format(avg_acc, loss.item()))
            for i in range(len(au_keys)):
                acc_in_epoch[i] = pred[i] + acc_in_epoch[i]
                total_in_epoch[i] = total[i] + total_in_epoch[i]
                if(total_in_epoch[i]!=0):
                    train_acc_in_epoch[i]=acc_in_epoch[i]/total_in_epoch[i]
                else:
                    train_acc_in_epoch[i]=0
    Aus_dict={}
    for i in range(len(au_keys)):
        Aus_dict.update({au_keys[i]:train_acc_in_epoch[i]})
    print(Aus_dict)
    train_loss = train_loss / batch_idx
    train_acc=sum(train_acc_in_epoch)/len(train_acc_in_epoch)
    print(train_acc)
    return train_loss,train_acc,Aus_dict
@torch.no_grad()
def test(epoch):
    print('\nEpoch(validation): %d' % epoch)
    net.eval()
    test_loss = 0
    au_keys = ['au1', 'au2', 'au4', 'au6', 'au7', 'au10', 'au12', 'au14', 'au15', 'au17', 'au23', 'au24']
    acc_in_epoch = [0 for i in range(len(au_keys))]
    total_in_epoch = [0 for i in range(len(au_keys))]
    test_acc_in_epoch = [0 for i in range(len(au_keys))]
    for batch_idx, (inputs, targets) in enumerate(tqdm(valloader)):
        inputs, targets = inputs.to(device), targets.to(device).squeeze(dim=1).float()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='mean').to(device)
        test_loss += loss.item()
        m = nn.Sigmoid()
        predicted = m(outputs)
        pred = [0 for i in range(len(au_keys))]  #一个batch的准确率总和
        for i in range(len(targets)):
            if (targets[i, :].nonzero().numel() != 0):
                index_1 = targets[i, :].nonzero()
                for j in index_1:
                    pred[j] = pred[j] + predicted[i, j].item()
        total = []
        total_acc = []  #一个batch里各个AU的总数, 一个batch里全部AU的平均准确率
        Sum = 0
        Not_nan = 0
        for i in range(len(au_keys)):
            total.append(targets[:, i].sum().item())
            if(total[i]!=0):
                acc = pred[i] / total[i]
                total_acc.append(acc)
            else:
                acc=0.5
                total_acc.append(acc)
            if(total_acc[i]!=0):
                Sum=total_acc[i]+Sum
                Not_nan=Not_nan+1
        if(Not_nan==0):
            print("This frame contains no AUs")             #里面有些数据没有标签 是否要去掉
        else:
            # avg_acc = Sum / Not_nan  #一个batch的平均准确率
            # print(str(batch_idx) + 'avg:{:.6f}|loss:{:.6f}'.format(avg_acc, loss.item()))
            for i in range(len(au_keys)):
                acc_in_epoch[i]=pred[i]+acc_in_epoch[i]     #整个验证集叠加的AU准确率
                if (total[i]!=0):
                    total_in_epoch[i] = total[i] + total_in_epoch[i] #整个验证集的AU数量
                    if(total_in_epoch!=0):
                        test_acc_in_epoch[i] = acc_in_epoch[i] / total_in_epoch[i]
                    else:
                        test_acc_in_epoch[i]=0
    Aus_dict = {}
    for i in range(len(au_keys)):
        Aus_dict.update({au_keys[i]: test_acc_in_epoch[i]})
    print(Aus_dict)
    test_loss = test_loss / batch_idx
    test_acc = sum(test_acc_in_epoch) / len(test_acc_in_epoch)
    print(test_acc)
    return test_loss, test_acc,Aus_dict

# -------------------------------------------------------------------------------
sequences, _ = BP4D_load_data.get_sequences_task()   # sequences, _文件名
train_seq, val_seq = get_train_val(sequences)   #MyDatasets.get_train_val()

if torch.cuda.is_available():
    deviceidx = [0, 1]
    device='cuda:1'
else:
    device='cpu'

# yes_no = input("Is this the first time running(yes/no):")
yes_no='yes'
print('==> Preparing data...')
transform_train = transforms.Compose([      #图片预处理  模型pretrained
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_val = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = MyBP4D(train_seq, train=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=180, shuffle=True, num_workers=0)

valset = MyBP4D(val_seq, train=False, transform=transform_val)
valloader = DataLoader(valset, batch_size=180, shuffle=True, num_workers=0)

#Model
print("start the net")
# net = ResNet34(12)
# net = Lnet(12)
net = Transformer(12)
# if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
#     net = torch.nn.DataParallel(net,deviceidx)    #多gpu训练,自动选择gpu
net.to(device)


train_lr = 0.001
optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=train_lr)
print('the learning rate is ', train_lr)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epoch =300
#断点重传
if(yes_no=="no"):
    start_epoch = -1
    print('-----------------------------')
    path_checkpoint ="./checkpoint/CHECKPOINT_FILE"
    checkpoint = torch.load(path_checkpoint)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("start_epoch:", start_epoch)
    print('-----------------------------')
    for epoch in range(start_epoch + 1, num_epoch):
        loss_dict = {}
        acc_dict = {}
        Aus_in_tra={}
        Aus_in_tes={}
        tra_loss,tra_acc,Aus_in_tra = train(epoch)
        tes_loss,tes_acc,Aus_in_tes = test(epoch)
        loss_dict.update({'train_loss': tra_loss})
        loss_dict.update({'test_loss': tes_loss})
        acc_dict.update({'train_acc': tra_acc})
        acc_dict.update({'test_acc': tes_acc})
        writer1.add_scalars('loss', loss_dict, global_step=epoch)
        writer1.add_scalars('acc', acc_dict, global_step=epoch)
        writer1.add_scalars('Aus_in_train', Aus_in_tra, global_step=epoch)
        writer1.add_scalars('Aus_in_test', Aus_in_tes, global_step=epoch)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, "./checkpoint/CHECKPOINT_FILE")
if(yes_no=="yes"):
    path_pretrain = "./Resnet34model/model_state.pth"
    pretrained = torch.load(path_pretrain)
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained)
    net.load_state_dict(model_dict)
    for epoch in range(start_epoch, start_epoch + num_epoch):
        loss_dict = {}
        acc_dict = {}
        Aus_in_tra = {}
        Aus_in_tes = {}
        tra_loss, tra_acc, Aus_in_tra = train(epoch)
        tes_loss, tes_acc, Aus_in_tes = test(epoch)
        loss_dict.update({'train_loss': tra_loss})
        loss_dict.update({'test_loss': tes_loss})
        acc_dict.update({'train_acc': tra_acc})
        acc_dict.update({'test_acc': tes_acc})
        writer1.add_scalars('loss', loss_dict, global_step=epoch)
        writer1.add_scalars('acc', acc_dict, global_step=epoch)
        writer1.add_scalars('Aus_in_train', Aus_in_tra, global_step=epoch)
        writer1.add_scalars('Aus_in_test', Aus_in_tes, global_step=epoch)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        torch.save(checkpoint, "./checkpoint/CHECKPOINT_FILE")







