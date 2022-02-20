from collections import OrderedDict
import os
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchmetrics.functional import ssim, psnr
from torchvision.transforms import Normalize, ToTensor, Compose, Resize, InterpolationMode
from src.datamodule import denormalize

from src.models.base import SuperResolutionModel
from src.models.core.spsr.SPSR_model import SPSRModel

opt = {
  'is_train':True,
  "name": "SPSR"   
  , "use_tb_logger": True
  , "model":"spsr"
  , "scale": 2
  , "gpu_ids": [0]
  , "network_G": {
    "which_model_G": "spsr_net" 
    , "norm_type": None
    ,"scale": 2
    , "mode": "CNA"
    , "nf": 64
    , "nb": 23
    , "in_nc": 3
    , "out_nc": 3
    , "gc": 32
    , "group": 1
  }

  , "network_D": {
    "which_model_D": "discriminator_vgg_128"
    , "norm_type": "batch"
    , "act_type": "leakyrelu"
    , "mode": "CNA"
    , "nf": 64
    , "in_nc": 3
  }

  , "train": {
    "lr_G": 1e-4
    , "lr_G_grad": 1e-4
    , "weight_decay_G": 0
    , "weight_decay_G_grad": 0
    , "beta1_G": 0.9
    , "beta1_G_grad": 0.9
    , "lr_D": 1e-4
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [50000, 100000, 200000, 300000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 1e-2
    , "feature_criterion": "l1"
    , "feature_weight": 1
    , "gan_type": "vanilla"
    , "gan_weight": 5e-3
    , "gradient_pixel_weight": 1e-2
    , "gradient_gan_weight": 5e-3
    , "pixel_branch_criterion": "l1"
    , "pixel_branch_weight": 5e-1
    , "Branch_pretrain" : 1
    , "Branch_init_iters" : 300

    , "manual_seed": 9
    , "niter": 5e5
    , "val_freq": 5e3
  }

  , "logger": {
    "print_freq": 100
    , "save_checkpoint_freq": 5e3
  }
}


class SPSR(SuperResolutionModel):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.cfg = config
        self.device = config.device

        self.means, self.stds = torch.Tensor(config.norm_means).to(
            self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.lr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])

        self._model = SPSRModel(opt)
        self.scale_factor = int(config.hr_img_size[0] / config.lr_img_size[0])

        self.global_step = 0

    def load(self):
        return NotImplemented

    @property
    def optimizers(self) -> List:
        return self._model.optimizers if self.training else []

    @property
    def schedulers(self) -> List:
        return self._model.schedulers if self.training else []

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.generator.device)
        if torch.is_floating_point(images):  # assumed that already normalized
            return images
        return self._prepr_op(images)

    def forward(self, inputs):
        return self._model.netG(inputs)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def training_step(self, batch):
        images = OrderedDict((
          ('LR', batch[0]),
          ('HR', batch[1]),
        ))
        self._model.feed_data(images)
        losses, fake_imgs = self._model.optimize_parameters(self.global_step)

        logs = OrderedDict()
        logs['losses'] = losses
        images['fake_H'] = fake_imgs[1]
        images['grad_LR'] = fake_imgs[2]
        images['G'] = fake_imgs[0]
        logs['images'] = images
        
        # logs = OrderedDict((
        #     ('losses', loss_dict),
        #     ('images', OrderedDict((
        #         ('LR', Xs),
        #         ('HR', Ys),
        #         ('U', sr_y),
        #     ))),
        # ))
        self.global_step += 1
        return logs

    @torch.no_grad()
    def eval_step(self, batch):
        x, y = batch[:2]
        fake_imgs = self(x)
        logs = OrderedDict((
            ('losses', OrderedDict()),
            ('metrics', OrderedDict((
                ('U_PSNR', psnr(fake_imgs[0], y)),
                ('U_SSIM', ssim(fake_imgs[0], y)),
            ))),
            ('images', OrderedDict((
                ('LR', x),
                ('HR', y),
                ('fake_H', fake_imgs[1]),
                ('grad_LR', fake_imgs[2]),
                ('G', fake_imgs[0]),
            ))),
        ))
        return logs
