import numpy as np
from scipy.stats import beta
from scipy.special import binom
import torch


def sample_mask(alpha=1., decay_power=3., shape=(256, 256), max_soft=0.0, reformulate=False):
    """Sample FMix mask.
    
    Args:
        alpha: Alpha parameter for beta distribution
        decay_power: Decay power for mask
        shape: Shape of the mask (H, W)
        max_soft: Maximum soft value
        reformulate: Whether to reformulate
        
    Returns:
        lam: Lambda value
        mask: Binary mask
    """
    lam = np.random.beta(alpha, alpha)
    H, W = shape
    
    # Sample mask
    mask = make_low_freq_image(decay_power, (H, W))
    mask = binarise_mask(mask, lam, (H, W), max_soft)
    
    return lam, mask


def make_low_freq_image(decay_power, shape):
    """Create low frequency image for FMix."""
    freqs = fftfreqnd(shape)
    spectrum = fftfreqnd(shape) ** decay_power
    spectrum[1:] = spectrum[1:] / spectrum[1:].max()
    spectrum[0] = 0
    
    spectrum = spectrum ** (1 / decay_power)
    spectrum[spectrum < 0.1] = 0
    
    mask = np.random.normal(size=shape, loc=0, scale=1)
    mask = np.fft.fftshift(np.fft.fftn(mask))
    mask *= spectrum
    mask = np.fft.ifftn(np.fft.ifftshift(mask)).real
    
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def fftfreqnd(shape):
    """Get FFT frequencies for n-dimensional shape."""
    freqs = []
    for dim in shape:
        freqs.append(np.fft.fftfreq(dim))
    return np.sqrt(np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.fftfreq(shape[1])[None, :] ** 2)


def binarise_mask(mask, lam, shape, max_soft=0.0):
    """Binarise mask based on lambda."""
    idx = mask.reshape(-1).argsort()[::-1]
    num = int(lam * mask.size)
    mask = np.zeros(mask.size)
    mask[idx[:num]] = 1
    
    if max_soft > 0:
        mask = mask.reshape(shape)
        mask = gaussian_blur(mask, max_soft)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    return mask.reshape(shape)


def gaussian_blur(mask, sigma):
    """Apply Gaussian blur to mask."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(mask, sigma=sigma)
