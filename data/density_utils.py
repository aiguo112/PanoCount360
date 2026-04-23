import cv2
import numpy as np

def resize_density_map_preserve_count(density: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = density.shape[:2]
    if in_h == out_h and in_w == out_w:
        return density.astype(np.float32)

    resized = cv2.resize(density, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    scale_factor = (in_h * in_w) / float(out_h * out_w)
    resized = resized * scale_factor
    return resized.astype(np.float32)