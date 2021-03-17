import torch
from torch.utils.data import Dataset

from PIL import Image
import random
import os
import BP4D_load_data

# Datasets

class MyDisfa(Dataset):
    # MyDataset继承Dataset, 重载了__init__, __getitem__, __len__
    def __init__(self, seq, train=True, transform=None, target_transform=None):
        self._seq = seq
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._train:
            self._train_data, self._train_labels = Disfa_load_data.integrate(self._seq)
            # print('the type of train_labels is ', type(self._train_labels))
        else:
            self._test_data, self._test_labels = Disfa_load_data.integrate(self._seq)
            # self._train_labels = self._train_labels[self._au]
            # self._test_labels = self._test_labels[self._au]

    def __getitem__(self, index):
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        image = Image.fromarray(image.reshape(240,240,3))
        # image = Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return image, target

    def __len__(self):
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

class MyBP4D(Dataset):
    def __init__(self, seq, train=True, transform=None, target_transform=None):
        self._seq = seq
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._train:
            self._train_data, self._train_labels = BP4D_load_data.load_data(self._seq)
        else:
            self._test_data, self._test_labels = BP4D_load_data.load_data(self._seq)

    def __getitem__(self, index):
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        image = Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return image, target

    def __len__(self):
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

class MyCK(Dataset):
    def __init__(self, seq, train=True, transform=None, target_transform=None):
        self._seq = seq
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._train:
            self._train_data, self._train_labels = CK_load_data.load_data(self._seq)
        else:
            self._test_data, self._test_labels = CK_load_data.load_data(self._seq)

    def __getitem__(self, index):
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        image = Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return image, target

    def __len__(self):
        if self._train:
            return len(self._train_data)
        return len(self._test_data)


def get_sequences(path):
    for root, dirs, files in os.walk(path):
        if len(dirs) > 1:
            return dirs

def get_test(seq, ratio=0.2):
    # 划分数据集
    total = len(seq)
    offset = int(total * ratio)
    random.shuffle(seq)
    train, test = seq[offset:], seq[:offset]
    return train, test

def get_train_val(seq, num_fold=6, fold=0):                  # cross_validation ? 验证集的前几个可能有空值了
    # 划分train和validation
    split = len(seq) // num_fold

    if fold == (num_fold - 1):
        train = seq[:fold*split]
        val = seq[fold*split:]
    else:
        train = seq[:fold*split] + seq[(fold+1)*split:]
        val = seq[fold*split : (fold+1)*split]
    return train, val



if __name__ == '__main__':
    sequences = [item for item in range(27)]
    temp_train, test = get_test(sequences)
    for fold in range(5):
        # train, test = get_test(sequences)
        # print(train, test)
        train, val = get_train_val(temp_train, fold=fold)
        print(train, val, test)
