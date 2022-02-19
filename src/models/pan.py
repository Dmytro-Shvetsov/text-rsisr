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


class PixelAttentionNetwork(SuperResolutionModel):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.cfg = config
        self.device = config.device

        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.lr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])
        self.scale_factor = int(config.hr_img_size[0] / config.lr_img_size[0])

        self.U = PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=self.scale_factor)
        self.opt_U = optim.Adam(self.U.parameters(), lr=config.lr)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=config.lr_milestones, gamma=0.9)

        self.global_step = 0
        self.l1_loss = nn.MSELoss()
        # self.gp_loss = GradientPriorLoss()

        self.pix_weight = 1
        # self.gp_weight = 1e-4
        self.gp_weight = 0

    def load(self):
        return NotImplemented
    
    @property
    def optimizers(self) -> List:
        return [self.opt_U] if self.training else []

    @property
    def schedulers(self) -> List:     
        return [self.lr_U] if self.training else []

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.generator.device)
        if torch.is_floating_point(images): # assumed that already normalized
            return images
        return self._prepr_op(images)
    
    def forward(self, inputs):
        return self.U(inputs)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def training_step(self, batch):
        Xs, Ys  = batch[:2]

        self.global_step += 1
        loss_dict = OrderedDict()

        # U
        self.opt_U.zero_grad()
        sr_y = self.U(Xs)
        loss_U_pix = self.l1_loss(sr_y, Ys)
        # loss_U_gp = self.gp_loss(sr_y, Ys)
        loss_U = self.pix_weight * loss_U_pix
        #  + self.gp_weight * loss_U_gp
        loss_U.backward()
        self.opt_U.step()
        loss_dict["U_pix"] = loss_U_pix.item()
        # loss_dict["U_gp"] = loss_U_gp.item()
        loss_dict["U"] = loss_U.item()

        logs = OrderedDict((
            ('losses', loss_dict),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('U', sr_y),
            ))),
        ))
        return logs

    @torch.no_grad()
    def eval_step(self, batch):
        Xs, Ys = batch[:2]

        sr = self.U(Xs)

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
