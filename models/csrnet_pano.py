"""
CSRNetPano: CSRNet for panoramic (equirectangular) crowd counting.
- Frontend: same as CSRNet (VGG pretrained).
- Backend: dilated convs with circular padding (ERP-aware).
- ERP-CBAM: latitude-aware channel + spatial attention after backend.
- LatitudePrior: learnable row-wise weights on density output.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csrnet import make_layers
from models.panocsrnet import make_layers_panocnn, LatitudePrior, CircularConv2d
from models.erp_cbam import ERP_CBAM


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: channel attention then spatial attention.
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        avg_out = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_out = F.adaptive_max_pool2d(x, 1).view(B, C)
        avg_out = self.fc(avg_out).view(B, C, 1, 1)
        max_out = self.fc(max_out).view(B, C, 1, 1)
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(cat))


class CSRNetPano(nn.Module):
    def __init__(self, out_h=64, load_pretrained_vgg=True):
        super().__init__()

        frontend_feat = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]
        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(
            frontend_feat, in_channels=3, batch_norm=False, dilation=False
        )
        self.backend = make_layers_panocnn(
            backend_feat, in_channels=512, dilation=True, circular=True
        )
        self.cbam = ERP_CBAM(64, reduction_ratio=16)
        # Multi-scale density head (Option A): fine + coarse branches then fuse
        self.head_fine = nn.Sequential(
            CircularConv2d(64, 32, kernel_size=3, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.head_coarse = nn.Sequential(
            CircularConv2d(64, 32, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.head_fuse = nn.Conv2d(2, 1, kernel_size=1)
        self.lat_prior = LatitudePrior(out_h=out_h)

        self._initialize_weights()

        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = x + self.cbam(x)  # residual around attention (Failure Mode C fix)
        fine = self.head_fine(x)
        coarse = self.head_coarse(x)
        x = self.head_fuse(torch.cat([fine, coarse], dim=1))
        x = self.lat_prior(x)
        x = torch.relu(x)
        return x

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
            for k in frontend_state.keys():
                if k in vgg_state and frontend_state[k].shape == vgg_state[k].shape:
                    matched[k] = vgg_state[k]

            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state)

            print("Loaded pretrained VGG16 weights into CSRNetPano frontend.")
        except Exception as e:
            print(f"Warning: could not load pretrained VGG16 weights: {e}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CSRNetPanoLatPriorOnly(nn.Module):
    """
    Phase 2.B: CSRNet + cosine latitude channel only (weight 0.1).
    No CBAM, no circular padding in decoder. Latitude channel concatenated to VGG features.
    """
    def __init__(self, out_h=64, load_pretrained_vgg=True):
        super().__init__()
        self.lat_weight = 0.1
        frontend_feat = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]
        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(
            frontend_feat, in_channels=3, batch_norm=False, dilation=False
        )
        self.backend = make_layers(
            backend_feat, in_channels=512 + 1, batch_norm=False, dilation=True
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        B, C, H, W = x.shape
        row = torch.arange(H, device=x.device, dtype=x.dtype)
        lat = self.lat_weight * torch.cos(torch.tensor(math.pi, device=x.device, dtype=x.dtype) * row / max(H - 1, 1))
        lat = lat.view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.cat([x, lat], dim=1)
        x = self.backend(x)
        x = self.output_layer(x)
        x = torch.relu(x)
        return x

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
            for k in frontend_state.keys():
                if k in vgg_state and frontend_state[k].shape == vgg_state[k].shape:
                    matched[k] = vgg_state[k]
            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state)
            print("Loaded pretrained VGG16 weights into CSRNetPanoLatPriorOnly frontend.")
        except Exception as e:
            print(f"Warning: could not load pretrained VGG16 weights: {e}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CSRNetPanoCircularOnly(nn.Module):
    """
    Phase 2.A: CSRNet with circular horizontal padding in decoder only.
    No attention (CBAM), no latitude prior. Same frontend as CSRNet, backend + head use circular padding.
    """
    def __init__(self, out_h=64, load_pretrained_vgg=True):
        super().__init__()
        frontend_feat = [
            64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512
        ]
        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(
            frontend_feat, in_channels=3, batch_norm=False, dilation=False
        )
        self.backend = make_layers_panocnn(
            backend_feat, in_channels=512, dilation=True, circular=True
        )
        self.head_fine = nn.Sequential(
            CircularConv2d(64, 32, kernel_size=3, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.head_coarse = nn.Sequential(
            CircularConv2d(64, 32, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.head_fuse = nn.Conv2d(2, 1, kernel_size=1)

        self._initialize_weights()
        if load_pretrained_vgg:
            self._load_pretrained_frontend()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        fine = self.head_fine(x)
        coarse = self.head_coarse(x)
        x = self.head_fuse(torch.cat([fine, coarse], dim=1))
        x = torch.relu(x)
        return x

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
            for k in frontend_state.keys():
                if k in vgg_state and frontend_state[k].shape == vgg_state[k].shape:
                    matched[k] = vgg_state[k]
            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state)
            print("Loaded pretrained VGG16 weights into CSRNetPanoCircularOnly frontend.")
        except Exception as e:
            print(f"Warning: could not load pretrained VGG16 weights: {e}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
