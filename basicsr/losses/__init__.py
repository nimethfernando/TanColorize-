# Import all losses to register them
from .losses import (
    L1Loss, MSELoss, CharbonnierLoss, WeightedTVLoss,
    PerceptualLoss, GANLoss, MultiScaleGANLoss, GANFeatLoss,
    ColorfulnessLoss
)

from basicsr.utils.registry import LOSS_REGISTRY


def build_loss(opt):
    """Build loss from registry.

    Args:
        opt (dict): Loss configuration. It must contain the key "type".

    Returns:
        Loss: The built loss.
    """
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    return loss
