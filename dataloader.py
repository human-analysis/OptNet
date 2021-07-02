# dataloader.py

import torch
import datasets
import torch.utils.data
import torchvision.transforms as transforms


class PrivacyDataLoader:
    def __init__(self, args, encoder=1):

        self.args = args
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train
        self.dataset_root_train = args.dataset_root_train
        self.dataset_root_test = args.dataset_root_test
        self.input_filename_train = args.input_filename_train
        self.input_filename_test = args.input_filename_test
        self.resolution = (args.resolution_wide, args.resolution_high)
        if encoder == 1:
            self.batch_size = self.args.batch_size_train
        else:
            self.batch_size = self.args.batch_size
        if self.dataset_train_name == 'CelebA_Privacy':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.dataset_root_train,
                ifile=self.input_filename_train,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    #transforms.CenterCrop(self.resolution),
                    # transforms.RandomCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.dataset_root_test,
                ifile=self.input_filename_test,
                transform=transforms.Compose([
                    #transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

            )

        elif self.dataset_train_name == 'Gaussian':
            self.dataset_train = datasets.Gaussian(
                self.args.dataroot, train=True)
            self.dataset_test = datasets.Gaussian(
                self.args.dataroot, train=False)

        elif self.dataset_train_name == 'GaussianNew':
            self.dataset_train = datasets.GaussianNew(
                self.args.dataroot, train=True)
            self.dataset_test = datasets.GaussianNew(
                self.args.dataroot, train=False)

    def create(self, flag=None):
        dataloader = {}
        if flag == "Train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )
            return dataloader

        elif flag == "Test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size_test,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True
            )
            return dataloader

        elif flag is None:
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size_test,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True)

            return dataloader