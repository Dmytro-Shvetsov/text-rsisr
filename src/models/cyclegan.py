from tkinter import Variable
from typing import Callable
import cv2
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import ssim, psnr
from torchvision.transforms import Normalize, Compose, ToTensor, Resize, InterpolationMode
from collections import OrderedDict

from src import models
from src.datamodule import denormalize
from src.models.core.cyclegan.models import GeneratorResNet, Discriminator
from src.models.core.cyclegan.utils import LambdaLR, ReplayBuffer
from src.utils.config_reader import Config, object_from_dict
from src.models.base import SuperResolutionModel
from src.datamodule import denormalize


class CycleGAN(SuperResolutionModel):
    def __init__(self, config:Config, **kwargs):
        super().__init__()

        self.cfg = config
        self.is_loaded = False

        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.lr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(config.norm_means, config.norm_stds),
        ])
        assert config.lr_img_size == config.hr_img_size
        input_shape = (3, *config.hr_img_size)

        # Initialize generator and discriminator
        self.G_AB = GeneratorResNet(input_shape, kwargs.get('n_residual_blocks', 9))
        self.G_BA = GeneratorResNet(input_shape, kwargs.get('n_residual_blocks', 9))
        self.D_A = Discriminator(input_shape)
        self.D_B = Discriminator(input_shape)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=config.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=config.lr, betas=(0.5, 0.999))

        # Learning rate update schedulers
        decay_epoch = config.num_epochs // 2
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(config.num_epochs, 0, decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(config.num_epochs, 0, decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(config.num_epochs, 0, decay_epoch).step
        )

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.alpha_cyc = 10.
        self.alpha_id = 5.

        self.fake_A_buffer = ReplayBuffer(50)
        self.fake_B_buffer = ReplayBuffer(50)

    def load(self):
        if self.is_loaded:
            return
        self.is_loaded = True
        return NotImplemented

    @property
    def optimizers(self):
        return [self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B] if self.training else []

    @property
    def schedulers(self):
        return [self.lr_scheduler_G, self.lr_scheduler_D_A, self.lr_scheduler_D_B] if self.training else []

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.G_AB.device)
        if torch.is_floating_point(images): # assumed that already normalized
            return images
        return self._prepr_op(images)

    def forward(self, x):
        return self.G_AB(x)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def generators_step(self, batch):
        real_A, real_B, _ = batch

        valid = torch.autograd.Variable(torch.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False).type_as(real_A)
        # valid = torch.ones((real_A.size(0), *self.D_A.output_shape), requires_grad=False).type_as(real_A)

        # Identity loss (learn not to change the input image when there is no need to)
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.G_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
        fake_A = self.G_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + self.alpha_cyc * loss_cycle + self.alpha_id * loss_identity

        logs = OrderedDict({
            'loss_G': loss_G,
            'loss_G_GAN': loss_GAN,
            'loss_G_cyc': loss_cycle,
            'loss_G_identity': loss_identity,
        })
        return loss_G, (fake_A, fake_B), logs

    def discriminator_A_step(self, batch, fake_imgs):
        real_A, real_B, _ = batch

        fake_A, _ = fake_imgs

        valid = torch.autograd.Variable(torch.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False).type_as(real_A)
        fake = torch.autograd.Variable(torch.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False).type_as(real_A)

        # Real loss
        loss_real = self.criterion_GAN(self.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
        loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        logs = OrderedDict({
            'loss_D_A': loss_D_A,
            'loss_D_A_real': loss_real,
            'loss_D_A_fake': loss_fake,
        })
        return loss_D_A, logs

    def discriminator_B_step(self, batch, fake_imgs):
        real_A, real_B, _ = batch

        _, fake_B = fake_imgs

        valid = torch.autograd.Variable(torch.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False).type_as(real_A)
        fake = torch.autograd.Variable(torch.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False).type_as(real_A)

        # Real loss
        loss_real = self.criterion_GAN(self.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
        loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        logs = OrderedDict({
            'loss_D_B': loss_D_B,
            'loss_D_real': loss_real,
            'loss_D_fake': loss_fake,
        })
        return loss_D_B, logs

    def training_step(self, batch):
        batch[0] = self.preprocess(batch[0])

        loss_G, fake_imgs, G_logs = self.generators_step(batch)
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        loss_D_A, D_A_logs = self.discriminator_A_step(batch, fake_imgs)
        self.optimizer_D_A.zero_grad()
        loss_D_A.backward()
        self.optimizer_D_A.step()

        loss_D_B, D_B_logs = self.discriminator_B_step(batch, fake_imgs)
        self.optimizer_D_B.zero_grad()
        loss_D_B.backward()
        self.optimizer_D_B.step()

        logs = OrderedDict({'losses': OrderedDict(), 'images': OrderedDict()})
        logs['losses'].update(G_logs)
        logs['losses']['loss_D'] = (loss_D_A + loss_D_B) / 2
        logs['losses'].update(D_A_logs)
        logs['losses'].update(D_B_logs)
        logs['images'] = OrderedDict((
            ('LR', batch[0]),
            ('HR', batch[1]),
            ('G_AB_images', fake_imgs[0]),
            ('G_BA_images', fake_imgs[1]),
        ))
        return logs

    @torch.no_grad()
    def eval_step(self, batch):
        batch[0] = self.preprocess(batch[0])
        x, y, _ = batch

        loss_G, fake_imgs, G_logs = self.generators_step(batch)
        loss_D_A, D_A_logs = self.discriminator_A_step(batch, fake_imgs)
        loss_D_B, D_B_logs = self.discriminator_B_step(batch, fake_imgs)
        
        logs = OrderedDict({'losses': OrderedDict(), 'images': OrderedDict()})
        logs['losses'].update(G_logs)
        logs['losses']['loss_D'] = (loss_D_A + loss_D_B) / 2
        logs['losses'].update(D_A_logs)
        logs['losses'].update(D_B_logs)
        logs['images'] = OrderedDict((
            ('LR', batch[0]),
            ('HR', batch[1]),
            ('G_AB_images', fake_imgs[0]),
            ('G_BA_images', fake_imgs[1]),
        ))

        logs['metrics'] = OrderedDict((
            ('G_AB_PSNR', psnr(fake_imgs[1], y)),
            ('G_AB_SSIM', ssim(fake_imgs[1], y)),
            ('G_BA_PSNR', psnr(fake_imgs[0], x)),
            ('G_BA_SSIM', ssim(fake_imgs[0], x)),
        ))
        return logs
