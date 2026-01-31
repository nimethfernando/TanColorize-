# Import all architectures to register them
from .tancolorize_arch import TanColorize
from .vgg_arch import VGGFeatureExtractor
from .discriminator_arch import DynamicUNetDiscriminator

from basicsr.utils.registry import ARCH_REGISTRY


def build_network(opt):
    """Build network from registry.

    Args:
        opt (dict): Network configuration. It must contain the key "type".

    Returns:
        nn.Module: The built network.
    """
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    return net
