from collections import OrderedDict
import os
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.functional import ssim, psnr
from torchvision.transforms import Normalize, ToTensor, Compose, Resize, InterpolationMode
from src.datamodule import denormalize

from src.models.base import SuperResolutionModel

from src.losses.gradient_prior import GradientPriorLoss
from .core.pan.models import PAN


class InterpolationModel(SuperResolutionModel):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.cfg = config
        self.device = config.device

        self.scale_factor = int(config.hr_img_size[0] / config.lr_img_size[0])
        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.lr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])
        self.inter_type = kwargs.get('mode')

    def load(self):
        return NotImplemented
    
    @property
    def optimizers(self) -> List:
        return []

    @property
    def schedulers(self) -> List:     
        return []

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.generator.device)
        if torch.is_floating_point(images): # assumed that already normalized
            return images
        return self._prepr_op(images)

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor, mode=self.inter_type)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def training_step(self, batch):
        pass

    @torch.no_grad()
    def eval_step(self, batch):
        Xs, Ys = batch[:2]

        sr = self(Xs)

        logs = OrderedDict((
            ('losses', OrderedDict()),
            ('metrics', OrderedDict((
                ('U_PSNR', psnr(sr, Ys)),
                ('U_SSIM', ssim(sr, Ys)),
            ))),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('U', sr),
            ))),
        ))
        return logs
