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

        # define network net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        
        # # PERFORMANCE OPTIMIZATION: torch.compile (requires PyTorch 2.0+)
        # # This can provide a 20-30% speedup by fusing kernels
        # if hasattr(torch, 'compile') and torch.cuda.is_available():
        #     try:
        #         self.net_g = torch.compile(self.net_g)
        #         get_root_logger().info("Model successfully compiled with torch.compile for speed.")
        #     except Exception as e:
        #         get_root_logger().warning(f"torch.compile failed, proceeding with standard mode: {e}")

        self.print_network(self.net_g)
        
        # load pretrained model for net_g
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
            self.use_amp = self.opt['train'].get('use_amp', False)
            # Mixed Precision setup
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        
        # # Discriminators benefit from compile too
        # if hasattr(torch, 'compile') and torch.cuda.is_available():
        #     self.net_d = torch.compile(self.net_d)

        self.print_network(self.net_d)

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        if train_opt.get('colorfulness_opt'):
            self.cri_colorfulness = build_loss(train_opt['colorfulness_opt']).to(self.device)
        else:
            self.cri_colorfulness = None

        self.setup_optimizers()
        self.setup_schedulers()

        # self.real_mu, self.real_sigma = None, None
        # if self.opt['val'].get('metrics') is not None and self.opt['val']['metrics'].get('fid') is not None:
        #     self._prepare_inception_model_fid()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params_g = self.net_g.parameters()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # Using zero_like tensors for RGB conversion to avoid allocation overhead
        self.lq_rgb = tensor_lab2rgb(torch.cat([self.lq, torch.zeros_like(self.lq), torch.zeros_like(self.lq)], dim=1))
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_lab = torch.cat([self.lq, self.gt], dim=1)
            self.gt_rgb = tensor_lab2rgb(self.gt_lab)

            if self.opt['train'].get('color_enhance', False):
                for i in range(self.gt_rgb.shape[0]):
                    self.gt_rgb[i] = color_enhacne_blend(self.gt_rgb[i], factor=self.opt['train'].get('color_enhance_factor'))

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            self.output_ab = self.net_g(self.lq_rgb)
            self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
            self.output_rgb = tensor_lab2rgb(self.output_lab)

            l_g_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output_ab, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output_rgb, self.gt_rgb)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            if self.cri_gan:
                fake_g_pred = self.net_d(self.output_rgb)
                l_g_gan = self.cri_gan(fake_g_pred, target_is_real=True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan
            # colorfulness loss
            if self.cri_colorfulness:
                l_g_color = self.cri_colorfulness(self.output_rgb)
                l_g_total += l_g_color
                loss_dict['l_g_color'] = l_g_color

        self.scaler.scale(l_g_total).backward()
        self.scaler.step(self.optimizer_g)

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            real_d_pred = self.net_d(self.gt_rgb)
            fake_d_pred = self.net_d(self.output_rgb.detach())
            l_d = self.cri_gan(real_d_pred, target_is_real=True, is_disc=True) + self.cri_gan(fake_d_pred, target_is_real=False, is_disc=True)
        
        loss_dict['l_d'] = l_d
        loss_dict['real_score'] = real_d_pred.detach().mean()
        loss_dict['fake_score'] = fake_d_pred.detach().mean()

        self.scaler.scale(l_d).backward()
        self.scaler.step(self.optimizer_d)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output_ab = self.net_g(self.lq_rgb)
            self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
            self.output_rgb = tensor_lab2rgb(self.output_lab)
        self.net_g.train()
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image') if use_pbar else None
        
        if self.opt['val']['metrics'] is not None and self.opt['val']['metrics'].get('fid') is not None:
            fake_acts_set, acts

def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output_ab = self.net_g(self.lq_rgb)
            self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
            self.output_rgb = tensor_lab2rgb(self.output_lab)
        self.net_g.train()


def save_training_images(self, current_iter):
        visuals = self.get_current_visuals()
        save_img_path = osp.join(self.opt['path']['visualization'], f'iter_{current_iter}.png')
        img_lq = tensor2img(self.lq_rgb[0]) 
        img_gt = tensor2img(visuals['gt_rgb'][0])
        img_res = tensor2img(visuals['result'][0])
        img_concat = np.concatenate((img_lq, img_gt, img_res), axis=1)
        imwrite(img_concat, save_img_path)