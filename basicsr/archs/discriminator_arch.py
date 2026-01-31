from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F

@ARCH_REGISTRY.register()
class DynamicUNetDiscriminator(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN).
    Args:
        nf (int): Channel number of intermediate features. Default: 64.
        n_channels (int): Channel number of input images. Default: 3.
        skip_connection (bool): Whether to use skip connections. Default: True.
    """

    def __init__(self, nf=64, n_channels=3, skip_connection=True):
        super(DynamicUNetDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        num_feat = nf
        num_in_ch = n_channels

        # Encoder
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        
        # Center
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False))

        # Decoder
        # If skip_connection is True, we concatenate, so input channels double for the "skip" part
        # x4 (upsampled) has 8*nf. x3 has 8*nf. Concat -> 16*nf.
        # x5 (upsampled) has 4*nf. x2 has 4*nf. Concat -> 8*nf.
        # x6 (upsampled) has 2*nf. x1 has 2*nf. Concat -> 4*nf.
        # x7 (upsampled) has 1*nf. x0 has 1*nf. Concat -> 2*nf.

        grad_factor = 2 if skip_connection else 1

        self.conv5 = norm(nn.Conv2d(num_feat * 8 * grad_factor, num_feat * 4, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 4 * grad_factor, num_feat * 2, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat * 2 * grad_factor, num_feat, 3, 1, 1, bias=False))
        
        self.conv8 = norm(nn.Conv2d(num_feat * grad_factor, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # Encoder
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)
        
        # Center
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)

        # Decoder
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        if self.skip_connection:
            x4 = torch.cat([x4, x3], 1)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        if self.skip_connection:
            x5 = torch.cat([x5, x2], 1)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        
        x6 = F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=False)
        if self.skip_connection:
            x6 = torch.cat([x6, x1], 1)
        x7 = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        
        x7 = F.interpolate(x7, scale_factor=2, mode='bilinear', align_corners=False) 
        if self.skip_connection:
            x7 = torch.cat([x7, x0], 1)
        x8 = F.leaky_relu(self.conv8(x7), 0.2, inplace=True)
        
        out = self.conv9(x8)
        return out
