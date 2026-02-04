import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.metrics.custom_fid import INCEPTION_V3_FID, get_activations, calculate_activation_statistics, calculate_frechet_distance
from basicsr.utils.color_enhance import color_enhacne_blend

@MODEL_REGISTRY.register()
class ColorModel(BaseModel):
    """Colorization model for single image colorization."""

    def __init__(self, opt):
        super(ColorModel, self).__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
            self.use_amp = self.opt['train'].get('use_amp', False)
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        self.net_g.train()
        self.net_d.train()

        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device) if train_opt.get('gan_opt') else None

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        self.optimizer_g = self.get_optimizer(train_opt['optim_g'].pop('type'), self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        self.optimizer_d = self.get_optimizer(train_opt['optim_d'].pop('type'), self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_rgb = tensor_lab2rgb(torch.cat([self.lq, torch.zeros_like(self.lq), torch.zeros_like(self.lq)], dim=1))
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_rgb = tensor_lab2rgb(torch.cat([self.lq, self.gt], dim=1))

    def optimize_parameters(self, current_iter):
        for p in self.net_d.parameters(): p.requires_grad = False
        self.optimizer_g.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            self.output_ab = self.net_g(self.lq_rgb)
            self.output_rgb = tensor_lab2rgb(torch.cat([self.lq, self.output_ab], dim=1))
            l_g_total = 0
            loss_dict = OrderedDict()
            
            # 1. Pixel Loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output_ab, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            
            # 2. Perceptual AND Style Loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output_rgb, self.gt_rgb)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # 3. GAN Loss
            if self.cri_gan:
                l_g_gan = self.cri_gan(self.net_d(self.output_rgb), target_is_real=True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

        self.scaler.scale(l_g_total).backward()
        self.scaler.step(self.optimizer_g)

        for p in self.net_d.parameters(): p.requires_grad = True
        self.optimizer_d.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            real_d_pred = self.net_d(self.gt_rgb)
            fake_d_pred = self.net_d(self.output_rgb.detach())
            l_d = self.cri_gan(real_d_pred, True, True) + self.cri_gan(fake_d_pred, False, True)
        
        loss_dict['l_d'] = l_d
        self.scaler.scale(l_d).backward()
        self.scaler.step(self.optimizer_d)
        self.scaler.update()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output_ab = self.net_g(self.lq_rgb)
            self.output_rgb = tensor_lab2rgb(torch.cat([self.lq, self.output_ab], dim=1))
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_rgb'] = self.lq_rgb.detach().cpu()
        out_dict['result'] = self.output_rgb.detach().cpu()
        out_dict['gt_rgb'] = self.gt_rgb.detach().cpu()
        return out_dict

    def save_training_images(self, current_iter):
        """Saves a snapshot of training progress."""
        visuals = self.get_current_visuals()
        save_img_path = osp.join(self.opt['path']['visualization'], f'iter_{current_iter}.png')
        img_lq = tensor2img(visuals['lq_rgb'][0]) 
        img_gt = tensor2img(visuals['gt_rgb'][0])
        img_res = tensor2img(visuals['result'][0])
        img_concat = np.concatenate((img_lq, img_gt, img_res), axis=1)
        imwrite(img_concat, save_img_path)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Method called during validation phase."""
        use_pbar = self.opt['val'].get('pbar', False)
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='batch')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            
            # Handle batch of image names if available
            if 'lq_path' in val_data:
                img_names = [osp.splitext(osp.basename(path))[0] for path in val_data['lq_path']]
            else:
                img_names = [f'val_{idx}_{i}' for i in range(val_data['lq'].size(0))]

            if save_img:
                batch_size = visuals['result'].size(0)
                for b in range(batch_size):
                    result_tensor = visuals['result'][b]
                    gt_tensor = visuals['gt_rgb'][b] if 'gt_rgb' in visuals else None
                    lq_tensor = visuals['lq_rgb'][b]
                    
                    sr_img = tensor2img(result_tensor)
                    gt_img = tensor2img(gt_tensor) if gt_tensor is not None else None
                    lq_img = tensor2img(lq_tensor)
                    
                    if gt_img is not None:
                        img_concat = np.concatenate((lq_img, gt_img, sr_img), axis=1)
                    else:
                        img_concat = np.concatenate((lq_img, sr_img), axis=1)

                    img_name = img_names[b]
                    save_img_path = osp.join(self.opt['path']['visualization'], f'val_{img_name}_{current_iter}.png')
                    imwrite(img_concat, save_img_path)

            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

    @master_only
    def save(self, epoch, current_iter):
        """Save networks and training state during training."""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)