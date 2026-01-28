# Import all metrics to register them
from .psnr_ssim import calculate_psnr, calculate_ssim

from basicsr.utils.registry import METRIC_REGISTRY


def calculate_metric(data, opt):
    """Calculate metric from registry.

    Args:
        data (dict): Data dictionary containing 'img' and optionally 'img2'.
        opt (dict): Metric configuration. It must contain the key "type".

    Returns:
        float: The calculated metric value.
    """
    metric_type = opt.get('type')
    if metric_type is None:
        raise ValueError('metric type must be specified in opt')
    
    # Handle built-in metrics
    if metric_type == 'calculate_psnr':
        return calculate_psnr(data['img'], data['img2'], crop_border=opt.get('crop_border', 0))
    elif metric_type == 'calculate_ssim':
        return calculate_ssim(data['img'], data['img2'], crop_border=opt.get('crop_border', 0))
    else:
        # Try to get from registry
        metric_func = METRIC_REGISTRY.get(metric_type)
        return metric_func(data, opt)
