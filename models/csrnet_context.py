import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GlobalContextBlock(nn.Module):
    """
    Lightweight global attention module
    """

    def __init__(self, channels):
        super().__init__()

        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.shape

        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attention = torch.bmm(q.permute(0, 2, 1), k)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(v, attention)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x

        return out


def make_layers(cfg, in_channels=3, dilation=False):

    layers = []
    d_rate = 2 if dilation else 1

    for v in cfg:

        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        else:
            conv = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate
            )

            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class CSRNetContext(nn.Module):

    def __init__(self, load_pretrained_vgg=True):
        super().__init__()

        frontend_feat = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512
        ]

        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(frontend_feat)
        self.backend = make_layers(backend_feat, in_channels=512, dilation=True)

        # NEW
        self.context = GlobalContextBlock(64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        if load_pretrained_vgg:
            self._load_pretrained_vgg()

    def forward(self, x):

        x = self.frontend(x)
        x = self.backend(x)

        # NEW
        x = self.context(x)

        x = self.output_layer(x)
        x = F.relu(x)

        return x

    def _load_pretrained_vgg(self):

        try:
            vgg = models.vgg16(pretrained=True)

            frontend_dict = self.frontend.state_dict()
            vgg_dict = vgg.features.state_dict()

            matched = {}

            for (k1, v1), (k2, v2) in zip(frontend_dict.items(), vgg_dict.items()):
                if v1.shape == v2.shape:
                    matched[k1] = v2

            frontend_dict.update(matched)
            self.frontend.load_state_dict(frontend_dict)

            print("Loaded pretrained VGG16 weights.")

        except Exception as e:
            print("Warning: could not load VGG weights:", e)

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)