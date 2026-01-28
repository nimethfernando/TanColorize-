# Import all models to register them
from .color_model import ColorModel

from basicsr.utils.registry import MODEL_REGISTRY


def build_model(opt):
    """Build model from registry.

    Args:
        opt (dict): Model configuration. It must contain the key "model_type".

    Returns:
        Model: The built model.
    """
    model_type = opt.get('model_type')
    if model_type is None:
        raise ValueError('model_type must be specified in opt')
    
    model = MODEL_REGISTRY.get(model_type)(opt)
    return model
