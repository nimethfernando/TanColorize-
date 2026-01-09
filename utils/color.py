import numpy as np
import cv2
import torch


def lab_to_rgb_tensor(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Convert batched LAB (L in [0,1], ab in [-1,1]) to RGB in [0,1].

    Args:
        L: Bx1xHxW
        ab: Bx2xHxW
    Returns:
        RGB: Bx3xHxW in [0,1]
    """
    device = L.device
    B, _, H, W = L.shape

    # Use uint8 path which is consistent across OpenCV builds:
    # L in [0,255], a/b in [0,255] with 128 offset.

    L_u8 = np.clip((L.detach().cpu().numpy() * 255.0), 0.0, 255.0).astype(np.uint8)
    ab_u8 = np.clip((ab.detach().cpu().numpy() * 128.0 + 128.0), 0.0, 255.0).astype(np.uint8)

    out = []
    for b in range(B):
        L_hw1 = L_u8[b].transpose(1, 2, 0)  # HxWx1
        ab_hw2 = ab_u8[b].transpose(1, 2, 0)  # HxWx2
        lab = np.concatenate([L_hw1, ab_hw2], axis=2)  # HxWx3 uint8

        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # returns uint8 [0,255]
        rgb01 = (rgb.astype(np.float32) / 255.0)
        out.append(torch.from_numpy(rgb01.transpose(2, 0, 1)))

    return torch.stack(out, dim=0).to(device)


