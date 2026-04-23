import torch
import torch.nn as nn


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    layers = []
    d_rate = 2 if dilation else 1

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, load_pretrained_vgg=True):
        super().__init__()

        frontend_feat = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]
        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(frontend_feat, in_channels=3, batch_norm=False, dilation=False)
        self.backend = make_layers(backend_feat, in_channels=512, batch_norm=False, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = torch.relu(x)
        return x

    def _load_pretrained_frontend(self):
        try:
            from torchvision import models

            try:
                # newer torchvision
                vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except AttributeError:
                # older torchvision
                vgg16 = models.vgg16(pretrained=True)

            frontend_state = self.frontend.state_dict()
            vgg_state = vgg16.features.state_dict()

            matched = {}
            for k in frontend_state.keys():
                if k in vgg_state and frontend_state[k].shape == vgg_state[k].shape:
                    matched[k] = vgg_state[k]

            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state)

            print("Loaded pretrained VGG16 weights into CSRNet frontend.")
        except Exception as e:
            print(f"Warning: could not load pretrained VGG16 weights: {e}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)