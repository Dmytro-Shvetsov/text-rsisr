from collections import OrderedDict
from typing import Callable, Dict, Tuple
import cv2
from sklearn.feature_extraction import image
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import ssim, psnr
from torchvision.transforms import Normalize, ToTensor, Compose, Resize, InterpolationMode

from src import models
from src.datamodule import denormalize
from src.models.core.esrgan.models import FeatureExtractor, GeneratorRRDB, Discriminator
from src.utils.config_reader import Config, object_from_dict
from src.models.base import SuperResolutionModel
from src.losses.gradient_prior import GradientPriorLoss


class ESRGAN(SuperResolutionModel):
    def __init__(self, config:Config, **kwargs):
        super().__init__()
        self.cfg = config
        self.is_loaded = False

        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.lr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])
        scale_factor = config.hr_img_size[0] / config.lr_img_size[0]
        self.generator = GeneratorRRDB(3, filters=64, num_res_blocks=kwargs.get('num_rrdb_blocks', 23), num_upsample=int(scale_factor) - 1)
        self.discriminator = Discriminator(input_shape=(3, *config.hr_img_size))
        self.feature_extractor = FeatureExtractor()
        # feature extractor stays in inference mode
        self.feature_extractor.eval()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

        # Losses
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.content_loss = nn.L1Loss()
        self.pixel_loss = nn.L1Loss()
        self.gp_loss = GradientPriorLoss()
        self.alpha_adv = 5e-3
        self.alpha_pixel = 1e-2
        self.alpha_gp = 1e-6
        self._warmup_iters = kwargs.get('warmup_iters', 500)

        self.global_step = 0

    @property
    def optimizers(self):
        return [self.optimizer_G, self.optimizer_D] if self.training else []
    
    @property
    def schedulers(self):
        return []

    def load(self):
        if self.is_loaded:
            return
        gen_ckpt, disc_ckpt = self.cfg.get('generator_ckpt'), self.cfg.get('discriminator_ckpt')
        if gen_ckpt:
            self.generator.load_state_dict(gen_ckpt)
        if disc_ckpt:
            self.discriminator.load_state_dict(disc_ckpt)
        self.is_loaded = True

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.generator.device)
        if torch.is_floating_point(images): # assumed that already normalized
            return images
        return self._prepr_op(images)

    def forward(self, x):
        return self.generator(x)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def train(self, mode: bool = True):
        ret = super().train(mode)
        self.feature_extractor.eval()
        return ret

    def generator_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        x, y = batch[:2]
        valid = torch.ones((x.size(0), *self.discriminator.output_shape), requires_grad=False).type_as(x)
        # self.generator.train()
        # print(self.generator.training)
        fake_imgs = self(x)

        pixel_loss = self.pixel_loss(fake_imgs, y)
        if self.global_step < self._warmup_iters:
            return pixel_loss, fake_imgs, {'loss_G_pixel': pixel_loss.item()}

        gp_loss = self.gp_loss(fake_imgs, y)

        # optimize the generator to make discriminator think the fake samples are real
        real_preds = self.discriminator(y).detach()
        fake_preds = self.discriminator(fake_imgs)
        adv_loss = self.adv_loss(fake_preds - real_preds.mean(0, keepdims=True), valid)

        # content loss based on the features from a pretrained network
        real_feats = self.feature_extractor(y).detach()
        fake_feats = self.feature_extractor(fake_imgs)
        content_loss = self.content_loss(fake_feats, real_feats)
        
        gen_loss = self.alpha_pixel * pixel_loss + self.alpha_gp * gp_loss + self.alpha_adv * adv_loss + content_loss

        logs = OrderedDict((
            ('loss_G', gen_loss.item()),
            ('loss_G_pixel', pixel_loss.item()),
            ('loss_G_gp', gp_loss.item()),
            ('loss_G_content', content_loss.item()),
        ))
        return gen_loss, fake_imgs, logs

    def discriminator_step(self, batch, fake_imgs) -> Tuple[torch.Tensor, Dict[str, float]]:
        x, y = batch[:2]

        pred_real = self.discriminator(y)
        pred_fake = self.discriminator(fake_imgs.detach())

        valid = torch.ones((x.size(0), *self.discriminator.output_shape), requires_grad=False).type_as(y)
        fake = torch.zeros((x.size(0),*self.discriminator.output_shape), requires_grad=False).type_as(x)

        # relativistic adversarial losses
        real_loss = self.adv_loss(pred_real - pred_fake.mean(0, keepdims=True), valid)
        fake_loss = self.adv_loss(pred_fake - pred_real.mean(0, keepdims=True), fake)

        # discriminator loss is the average of these
        disc_loss = (fake_loss + real_loss) / 2

        logs = OrderedDict((
            ('loss_D', disc_loss.item()),
            ('loss_D_fake', fake_loss.item()),
            ('loss_D_real', real_loss.item()),
        ))
        return disc_loss, logs

    def training_step(self, batch):
        batch[0] = self.preprocess(batch[0])

        gen_loss, fake_imgs, gen_logs = self.generator_step(batch)
        self.optimizer_G.zero_grad()
        gen_loss.backward()
        self.optimizer_G.step()

        logs = OrderedDict((
            ('losses', gen_logs.copy()),
            ('images', OrderedDict((
                ('LR', batch[0]),
                ('HR', batch[1]),
                ('G', fake_imgs),
            ))),
        ))

        if self.global_step >= self._warmup_iters:
            disc_loss, disc_logs = self.discriminator_step(batch, fake_imgs)
            self.optimizer_D.zero_grad()
            disc_loss.backward()
            self.optimizer_D.step()
            logs['losses'].update(disc_logs)
        self.global_step += 1
        return logs

    @torch.no_grad()
    def eval_step(self, batch):
        batch[0] = self.preprocess(batch[0])
        _, y, _ = batch

        _, fake_imgs, gen_logs = self.generator_step(batch)

        logs = OrderedDict((
            ('losses', gen_logs.copy()),
            ('images', OrderedDict((
                ('LR', batch[0]),
                ('HR', batch[1]),
                ('G', fake_imgs),
            ))),
        ))

        if self.global_step >= self._warmup_iters:
            _, disc_logs = self.discriminator_step(batch, fake_imgs)
            logs['losses'].update(disc_logs)
        
        impsnr = psnr(fake_imgs, y)
        imssim = ssim(fake_imgs, y)
        
        logs['metrics'] = OrderedDict((
            ('PSNR', impsnr),
            ('SSIM', imssim)
        ))
        return logs
