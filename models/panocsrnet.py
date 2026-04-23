import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv2d(nn.Module):
    """
    Circular padding in width (ERP wrap-around), zero padding in height.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        self.pad_h = ((kh - 1) // 2) * dilation
        self.pad_w = ((kw - 1) // 2) * dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kh, kw),
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0.0)
        return self.conv(x)


def make_layers_panocnn(cfg, in_channels=3, dilation=False, circular=False):
    layers = []
    d_rate = 2 if dilation else 1

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            if circular:
                conv = CircularConv2d(
                    in_channels, v, kernel_size=3, dilation=d_rate, bias=True
                )
            else:
                conv = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate, bias=True
                )
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class LatitudePrior(nn.Module):
    """
    Learnable row-wise prior applied to density map.
    Shape: [1, 1, H, 1]
    """
    def __init__(self, out_h):
        super().__init__()
        self.row_weights = nn.Parameter(torch.ones(1, 1, out_h, 1))

    def forward(self, x):
        return x * self.row_weights


class PanoCSRNet(nn.Module):
    def __init__(self, out_h=64, load_pretrained_vgg=True):
        super().__init__()

        frontend_feat = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]
        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers_panocnn(
            frontend_feat, in_channels=3, dilation=False, circular=True
        )
        self.backend = make_layers_panocnn(
            backend_feat, in_channels=512, dilation=True, circular=True
        )

        self.head_fine = nn.Sequential(
            CircularConv2d(64, 32, kernel_size=3, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.head_coarse = nn.Sequential(
            CircularConv2d(64, 32, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.fuse = nn.Conv2d(2, 1, kernel_size=1)
        self.lat_prior = LatitudePrior(out_h=out_h)

        self._initialize_weights()

        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)

        d1 = self.head_fine(x)
        d2 = self.head_coarse(x)

        out = torch.cat([d1, d2], dim=1)
        out = self.fuse(out)
        out = self.lat_prior(out)
        out = torch.relu(out)
        return out

    def _load_pretrained_frontend(self):
        try:
            from torchvision import models
            try:
                vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except AttributeError:
                vgg16 = models.vgg16(pretrained=True)

            frontend_state = self.frontend.state_dict()
            vgg_state = vgg16.features.state_dict()

            matched = {}
            frontend_keys = list(frontend_state.keys())
            vgg_keys = list(vgg_state.keys())

            j = 0
            for k in frontend_keys:
                if "conv.weight" in k or "conv.bias" in k:
                    while j < len(vgg_keys):
                        vk = vgg_keys[j]
                        if frontend_state[k].shape == vgg_state[vk].shape:
                            matched[k] = vgg_state[vk]
                            j += 1
                            break
                        j += 1

            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state, strict=False)
            print("Loaded pretrained VGG16 weights into PanoCSRNet frontend.")
        except Exception as e:
            print(f"Warning: could not load pretrained VGG16 weights into PanoCSRNet: {e}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)