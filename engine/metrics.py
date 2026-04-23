import torch


@torch.no_grad()
def compute_count_errors(pred_density, gt_density):
    pred_count = pred_density.sum(dim=(1, 2, 3))
    gt_count = gt_density.sum(dim=(1, 2, 3))

    abs_err = torch.abs(pred_count - gt_count)
    sq_err = (pred_count - gt_count) ** 2

    return abs_err, sq_err, pred_count, gt_count