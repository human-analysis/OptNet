import torch
from torch.utils.data import Dataset
import scipy.io as sio

import config
import losses
import os

import copy
import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

import csv
import math
from random import shuffle
import datasets.loaders as loaders

class CelebA_Privacy(Dataset):
    def __init__(self, ifile, root=None, split=1.0,
                 transform=None, loader='loader_image',
                 prefetch=False):

        self.root = root
        self.ifile = ifile
        self.split = split
        self.prefetch = prefetch
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        # import pdb; pdb.set_trace()
        self.nattributes = 0
        datalist = []
        classname = []
        if ifile is not None:
            with open(ifile, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if self.nattributes <= len(row):
                        self.nattributes = len(row)
                    datalist.append(row)
                    classname.append(row[1])
            csvfile.close()
        else:
            datalist = []

        if (self.split < 1.0) & (self.split > 0.0):
            if len(datalist) > 0:
                datalist = shuffle(datalist, classname)
                num = math.floor(self.split * len(datalist))
                self.data = datalist[0:num]
            else:
                self.data = []

        elif self.split == 1.0:
            if len(datalist) > 0:
                self.data = datalist
            else:
                self.data = []

        self.classname = list(set(classname))
        self.classname.sort()

        if prefetch is True:
            print('Prefetching data, feel free to stretch your legs !!')
            self.objects = []
            for index in range(len(self.data)):
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                self.objects.append(self.loader(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) > 0:
            if self.prefetch is False:
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                image = self.loader(path)
            elif self.prefetch is True:
                image = self.objects[index]

            attributes = self.data[index]
            fmetas = [attributes[0], attributes[1]]
            attributes = attributes[1:]
            attributes[0] = self.classname.index(attributes[0])
            if len(attributes) > 1:
                try:
                    attributes[1:] = [int(x) for x in attributes[1:]]
                except:
                    attributes[1:] = [0 for x in attributes[1:]]
            else:
                attributes[1:] = [0 for x in range(self.nattributes - 2)]

        if self.transform is not None:
            image = self.transform(image)

        attributes = torch.Tensor(attributes)
        T = attributes[20]
        S = attributes[32]

        return image, T, S

# 0 is person PID train_bal, test_bal,
# 1  5_o_Clock_Shadow      88.83, 90.01
# 2  Arched_Eyebrows       73.41, 71.55
# 3  Attractive            51.36, 50.41  ***
# 4  Bags_Under_Eyes       79.55, 79.74
# 5  Bald                  97.72, 97.88
# 6  Bangs                 84.83, 84.42
# 7  Big_Lips              75.91, 67.30
# 8  Big_Nose              76.44, 78.80
# 9  Black_Hair            76.10, 72.84
# 10  Blond_Hair           85.10, 86.67
# 11  Blurry               94.86, 94.94
# 12  Brown_Hair           79.61, 82.04
# 13  Bushy_Eyebrows       85.63, 87.04
# 14  Chubby               94.23, 94.70
# 15  Double_Chin          95.35, 95.43
# 16  Eyeglasses           93.54, 93.55
# 17  Goatee               93.65, 95.42
# 18  Gray_Hair            95.76, 96.81
# 19  Heavy_Makeup         61.57, 59.50  **
# 20  High_Cheekbones      54.76, 51.82  ***
# 21  Male                 58.06, 61.35  ***
# 22  Mouth_Slightly_Open  51.78, 50.49  ***
# 23  Mustache             95.92, 96.13
# 24  Narrow_Eyes          88.41, 85.13
# 25  No_Beard             83.42, 85.37
# 26  Oval_Face            71.68, 70.44
# 27  Pale_Skin            95.70, 95.80
# 28  Pointy_Nose          72.45, 71.42
# 29  Receding_Hairline    91.99, 91.51
# 30  Rosy_Cheeks          99.53, 92.83
# 31  Sideburns            94.37, 95.36
# 32  Smiling              52.03, 50.03  ***
# 33  Straight_Hair        79.14, 79.01
# 34  Wavy_Hair            68.06, 63.59  *
# 35  Wearing_Earrings     81.35, 79.33
# 36  Wearing_Hat          95.06, 95.80
# 37  Wearing_Lipstick     53.04, 52.19 ***
# 38  Wearing_Necklace     87.86, 86.21
# 39  Wearing_Necktie      92.70, 92.99
# 40  Young                77.89, 75.71

# (19, 20):  0.0729, 0.0951 *
# (19, 21):  0.4439, 0.4227 ****
# (19, 22):  0.0106, 0.0130
# (19, 32):  0.0308, 0.0382
# (19, 34):  0.1041, 0.1072 **
# (19, 37):  0.6434, 0.5791 ****

# (20, 21):  0.0615, 0.0781
# (20, 22):  0.1747, 0.1741 ***
# (20, 32):  0.4662, 0.4582 ****
# (20, 34):  0.0131, 0.0130
# (20, 37):  0.0793, 0.0936

# First: (20, 32)

class Gaussian(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        if train:

            D_train = sio.loadmat(root+'D_train.mat')
            D = torch.from_numpy(D_train['D_train_CVPR'])
            self.X = torch.t(D[0:2,:]).float()
            self.Y = torch.t(D[2:3,:])
            self.S = torch.t(D[3:4,:]).float()
            # import pdb
            # pdb.set_trace()
            # self.X, theta = L_SARL_train(self.X, self.S, self.Y, args.r, args.lambd, "cpu")

        else:
            D_test = sio.loadmat(root + 'D_test.mat')
            D = torch.from_numpy(D_test['D_test_CVPR'])
            self.X = torch.t(D[0:2,:]).float()
            self.Y = torch.t(D[2:3,:])
            self.S = torch.t(D[3:4,:]).float()

    def __len__(self):
            return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        return X, X, S


class GaussianNew(Dataset):

    def __init__(self, root, train):
        if train:

            data_train = np.load(root + 'data_train.npy', allow_pickle=True)
            data = torch.from_numpy(data_train).float()
            self.y = data[:, 0:2]
            # self.y[:, 1] = torch.pow(data[:, 1], 3)
            self.label = data[:, 2].long()
            self.s = data[:, 0]
            # self.s = torch.pow(data[:, 0].unsqueeze(1), 3)
            # self.s = torch.cos(np.pi * data[:, 0] + np.pi / 2 * torch.ones(data[:, 0].shape)).unsqueeze(1)
            # self.s = torch.sin(np.pi * data[:, 0] + np.pi / 2 * torch.ones(data[:, 0].shape)).unsqueeze(1)

            data_train1 = np.load(root + 'data_train.npy', allow_pickle=True)
            data1 = torch.from_numpy(data_train1).float()
            self.x = data1[:, 0:1]
            # self.x[:, 0] = 1 * torch.sin(np.pi * data1[:, 0] + np.pi / 2 * torch.ones(data1[:, 0].shape))
            # self.x[:, 0] = 1 * torch.cos(np.pi * data1[:, 0] + np.pi / 2 * torch.ones(data1[:, 0].shape))
            # self.x[:, 1] = 0.5 * torch.sin(np.pi * data1[:, 1] + np.pi / 2 * torch.ones(data1[:, 1].shape))
            # self.x[:, 1] = torch.pow(data1[:, 1], 3)

        else:
            data_test = np.load(root + 'data_test.npy', allow_pickle=True)
            data = torch.from_numpy(data_test).float()
            self.y = data[:, 0:2]
            # self.y[:, 1] = torch.pow(data[:, 1], 3)
            self.label = data[:, 2].long()
            self.s = data[:, 0]
            # self.s = torch.pow(data[:, 0].unsqueeze(1), 3)
            # self.s = torch.cos(np.pi * data[:, 0] + np.pi / 2 * torch.ones(data[:, 0].shape)).unsqueeze(1)
            # self.s = torch.sin(np.pi * data[:, 0] + np.pi / 2 * torch.ones(data[:, 0].shape)).unsqueeze(1)

            data_test1 = np.load(root + 'data_test.npy', allow_pickle=True)
            data1 = torch.from_numpy(data_test1).float()
            self.x = data1[:, 0:1]
            # self.x[:, 0] = 1 * torch.sin(np.pi * data1[:, 0] + np.pi / 2 * torch.ones(data1[:, 0].shape))
            # self.x[:, 0] = 1 * torch.cos(np.pi * data1[:, 0] + np.pi / 2 * torch.ones(data1[:, 0].shape))
            # self.x[:, 1] = 0.5 * torch.sin(np.pi * data1[:, 1] + np.pi / 2 * torch.ones(data1[:, 1].shape))
            # self.x[:, 1] = torch.pow(data1[:, 1], 3)

    def __len__(self):
            return len(self.x)

    def __getitem__(self, index):
        x, y, s = self.x[index], self.y[index], self.s[index]
        return x, y, s

class Adult(Dataset):

        def __init__(self, root, train=True,
                     transform=None):
            # import pdb; pdb.set_trace()
            self.data = sio.loadmat(root+'data.mat')
            if train:
                X = torch.from_numpy(self.data['X'])
                Y = torch.from_numpy(self.data['Y']).squeeze()
                S = torch.from_numpy(self.data['S']).squeeze()

                self.X = torch.t(X)
                self.Y = torch.t(Y)
                self.S = torch.t(S)
                # import pdb
                # pdb.set_trace()

            else:
                X = torch.from_numpy(self.data['X_test'])
                Y = torch.from_numpy(self.data['Y_test']).squeeze()
                S = torch.from_numpy(self.data['S_test']).squeeze()

                self.X = torch.t(X)
                self.Y = torch.t(Y)
                self.S = torch.t(S)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            X, Y, S = self.X[index], self.Y[index], self.S[index]
            return X, Y, S


class IMDB_Privacy(Dataset):
    def __init__(self, ifile, root=None,
                 transform=None, loader='loader_image',
                 prefetch=True, train=True):

        self.root = root
        self.ifile = ifile
        self.prefetch = prefetch
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        self.nattributes = 0
        datalist = []
        ages = []
        if ifile is not None:
            with open(ifile, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if self.nattributes <= len(row):
                        self.nattributes = len(row)
                    datalist.append(row)
                    ages.append(row[1])
            csvfile.close()
        else:
            datalist = []

        if len(datalist) > 0:
            self.data = datalist[1:]
        else:
            self.data = []

        if self.prefetch is True:
            print('Prefetching data, feel free to stretch your legs !!')
            self.objects = []
            for index in range(len(self.data)):
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                self.objects.append(self.loader(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) > 0:
            if self.prefetch is False:
                if self.root is not None:
                    path = os.path.join(self.root, self.data[index][0])
                else:
                    path = self.data[index][0]
                image = self.loader(path)
            elif self.prefetch is True:
                image = self.objects[index]

#             attributes = self.data[index]
#             fmetas = [attributes[0], attributes[1]]
#             attributes = attributes[1:]
#             attributes[0] = self.classname.index(attributes[0])
#             if len(attributes) > 1:
#                 try:
#                     attributes[1:] = [int(x) for x in attributes[1:]]
#                 except:
#                     attributes[1:] = [0 for x in attributes[1:]]
#             else:
#                 attributes[1:] = [0 for x in range(self.nattributes - 2)]

        if self.transform is not None:
            image = self.transform(image)



        # attributes = torch.Tensor(attributes)
        # S = float(self.data[index][3]) # age
        # T = float(self.data[index][2])# gender

        S = float(self.data[index][2]) # gender
        T = float(self.data[index][3])# age

        return image, T, S


class IMDB_sphere(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        if train:
            D = torch.load(root+'X_train.pt')
            Y = torch.load(root+'Y_train.pt')
            S = torch.load(root+'S_train.pt')

            self.X = D[0:-2,:]
            # self.X = self.X / torch.max(self.X)
            self.X = self.X / (torch.norm(self.X, dim=1)[:, None] + 1e-16)
            self.Y = Y[0:-2]
            self.S = S[0:-2]

        else:
            D = torch.load(root+'X_test.pt')
            Y = torch.load(root+'Y_test.pt')
            S = torch.load(root+'S_test.pt')

            self.X = D[0:-2,:]
            # self.X = self.X / torch.max(self.X)
            self.X = self.X / (torch.norm(self.X, dim=1)[:, None] + 1e-16)
            self.Y = Y[0:-2]
            self.S = S[0:-2]


    def __len__(self):
            return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index], self.S[index]
        # S and Y are exchanged
        return X, Y, S
        # return X, S, Y
