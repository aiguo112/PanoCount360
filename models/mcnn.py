import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


class MCNNBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch, k, p, use_pool in cfg:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p))
            layers.append(nn.ReLU(inplace=True))
            if use_pool:
                layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # All branches output at 1/4 resolution
        self.branch1 = MCNNBranch([
            (16, 9, 4, True),
            (32, 7, 3, True),
            (16, 7, 3, False),
            (8, 7, 3, False),
        ])

        self.branch2 = MCNNBranch([
            (20, 7, 3, True),
            (40, 5, 2, True),
            (20, 5, 2, False),
            (10, 5, 2, False),
        ])

        self.branch3 = MCNNBranch([
            (24, 5, 2, True),
            (48, 3, 1, True),
            (24, 3, 1, False),
            (12, 3, 1, False),
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(30, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)