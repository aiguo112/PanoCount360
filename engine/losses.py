import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_density_preserve_count(density, target_h, target_w):
    """
    density: [B, 1, H, W]
    returns resized density with approximately preserved sum
    """
    b, c, h, w = density.shape
    if h == target_h and w == target_w:
        return density

    resized = F.interpolate(
        density,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False
    )

    scale = (h * w) / float(target_h * target_w)
    resized = resized * scale
    return resized


class DensityCountLoss(nn.Module):
    def __init__(self, density_weight=1.0, count_weight=0.0):
        super().__init__()
        self.density_weight = density_weight
        self.count_weight = count_weight
        self.density_criterion = nn.MSELoss()

    def forward(self, pred_density, gt_density):
        _, _, ph, pw = pred_density.shape
        gt_density_resized = resize_density_preserve_count(gt_density, ph, pw)

        density_loss = self.density_criterion(pred_density, gt_density_resized)

        pred_count = pred_density.sum(dim=(1, 2, 3))
        gt_count = gt_density.sum(dim=(1, 2, 3))
        count_loss = torch.mean(torch.abs(pred_count - gt_count))

        total_loss = self.density_weight * density_loss + self.count_weight * count_loss

        return {
            "loss": total_loss,
            "density_loss": density_loss.detach(),
            "count_loss": count_loss.detach()
        }