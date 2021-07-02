import torch.nn as nn
import config
import torch.nn.functional as F
import torch

args = config.parse_args()

__all__ = ['E_Gaussian', 'EA', 'ET', 'Adversary', 'Target', 'Adversary_Gaussian', 'Target_Gaussian']


class E_Gaussian(nn.Module):
    def __init__(self, indim =args.ndim, r=args.r, hdlayers=args.hdlayers):
        super().__init__()
        self.model1 = nn.Sequential(
        nn.Linear(indim, hdlayers),
        nn.PReLU(),
        # nn.Dropout(0.5),
        nn.Linear(hdlayers, int(hdlayers/2)),
        nn.PReLU(),
        )
        self.classlayer = nn.Linear(int(hdlayers/2), r)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # out = out / (torch.norm(out, dim=1)[:, None] + 1e-16)
        return out

class EA(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_a, hdlayers=args.hdlayers):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, hdlayers),
            nn.PReLU(),
            nn.Linear(hdlayers, int(hdlayers/2)),
            nn.PReLU(),
        )
        self.classlayer = nn.Linear(int(hdlayers/2), num_classes)
        # self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # prob = self.softmaxlayer(out) + 1e-16
        return out

class ET(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_t, hdlayers=args.hdlayers):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, hdlayers),
            nn.PReLU(),
            nn.Linear(hdlayers, int(hdlayers/2)),
            nn.PReLU(),
        )
        self.classlayer = nn.Linear(int(hdlayers/2), num_classes)
        # self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # prob = self.softmaxlayer(out) + 1e-16
        return out


class Adversary_Gaussian(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_a):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 4),
            nn.PReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4, 4),
            nn.PReLU(),
            # nn.Linear(4, 2),
            # nn.ReLU(),
        )
        self.classlayer = nn.Linear(4, num_classes)
        # self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # prob = self.softmaxlayer(out) + 1e-16
        return out



class Target_Gaussian(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_t):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 4),
            nn.PReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4, 4),
            nn.PReLU(),
            # nn.Linear(4, 2),
            # nn.ReLU(),
        )
        self.classlayer = nn.Linear(4, num_classes)
        # self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # prob = self.softmaxlayer(out) + 1e-16
        return out


class Adversary(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_a):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 32),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.PReLU(),
        )
        self.classlayer = nn.Linear(16, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # prob = self.softmaxlayer(out) + 1e-16
        return out


class Target(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_t):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classlayer = nn.Linear(16, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        # prob = self.softmaxlayer(out) + 1e-16
        return out
