import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3


class INCEPTION_V3_FID(nn.Module):
    """Inception V3 model for FID calculation."""
    
    BLOCK_INDEX_BY_DIM = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3
    }

    def __init__(self, state_dict, block_indices):
        super(INCEPTION_V3_FID, self).__init__()
        self.model = inception_v3(pretrained=False, transform_input=False)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.fc = nn.Identity()
        self.model.aux_logits = False
        
        self.block_indices = block_indices
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Normalize
        x = (x - self.mean) / self.std
        
        # Resize to 299x299 if needed
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Forward through Inception
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x


def get_activations(images, model, batch_size=1):
    """Get activations from Inception model.
    
    Args:
        images: Tensor of images (B, C, H, W) with values in [0, 1]
        model: Inception model
        batch_size: Batch size for processing
        
    Returns:
        numpy array of activations
    """
    model.eval()
    activations = []
    
    with torch.no_grad():
        # Ensure images are in [0, 1] range
        if images.max() > 1.0:
            images = images / 255.0
        
        # Process in batches
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            act = model(batch)
            activations.append(act.cpu().numpy())
    
    return np.concatenate(activations, axis=0)


def calculate_activation_statistics(activations):
    """Calculate mean and covariance of activations.
    
    Args:
        activations: numpy array of activations
        
    Returns:
        mu: mean vector
        sigma: covariance matrix
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Distance between two multivariate Gaussians.
    
    Args:
        mu1: mean of first distribution
        sigma1: covariance of first distribution
        mu2: mean of second distribution
        sigma2: covariance of second distribution
        eps: small value for numerical stability
        
    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
