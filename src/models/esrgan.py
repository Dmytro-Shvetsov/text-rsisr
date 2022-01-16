from typing import Callable
import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import ssim, psnr

from src import models
from src.datamodule import denormalize
from src.models.core.esrgan.models import FeatureExtractor, GeneratorRRDB, Discriminator
from src.utils.config_reader import Config, object_from_dict
from src.models.base import SuperResolutionModelInterface
from torchvision.transforms import Normalize


class ESRGAN(pl.LightningModule, SuperResolutionModelInterface):
    def __init__(self, config:Config, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore='kwargs')
        self.cfg = config
        self._is_loaded = False

        self.normalize = Normalize(config.norm_means, config.norm_stds)
        self.generator = GeneratorRRDB(3, filters=64, num_res_blocks=kwargs.get('num_rrdb_blocks', 23), num_upsample=config.scale_factor - 1)
        self.discriminator = Discriminator(input_shape=(3, *config.hr_img_size))
        self.feature_extractor = FeatureExtractor()
        # feature extractor stays in inference mode
        self.feature_extractor.eval()
        self.load()

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.content_loss = nn.L1Loss()
        self.pixel_loss = nn.L1Loss()
        self.alpha_adv = 5e-3
        self.alpha_pixel = 1e-2
        self._warmup_iters = kwargs.get('warmup_iters', 500)
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def load(self):
        if self._is_loaded:
            return
        gen_ckpt, disc_ckpt = self.cfg.get('generator_ckpt'), self.cfg.get('discriminator_ckpt')
        if gen_ckpt:
            self.generator.load_state_dict(gen_ckpt)
        if disc_ckpt:
            self.discriminator.load_state_dict(disc_ckpt)
        self._is_loaded = True

    def preprocess(self, images):
        return images

    def forward(self, x):
        return self.generator(x)

    def parse_outputs(self, outputs):
        return outputs.int()

    def configure_optimizers(self):
        models = self.generator, self.discriminator
        opts = [object_from_dict(optc, params=m.parameters()) for optc, m in zip(self.cfg.optimizers, models)]
        schs = [object_from_dict(schc, optimizer=opt) for schc, opt in zip(self.cfg.schedulers, opts)]
        return opts, schs

    def train(self, mode: bool = True):
        ret = super().train(mode)
        self.feature_extractor.eval()
        return ret

    def generator_step(self, batch):
        x, y, _ = batch
        valid = torch.ones((x.size(0), *self.discriminator.output_shape), requires_grad=False).type_as(x)
        # self.generator.train()
        # print(self.generator.training)
        fake_imgs = self(x)
        # 
        # print(x.max(), y.max(), fake_imgs.max())
        pixel_loss = self.pixel_loss(fake_imgs, y)
        if self.global_step >= self._warmup_iters:
            return pixel_loss, fake_imgs

        # optimize the generator to make discriminator think the fake samples are real
        real_preds = self.discriminator(y).detach()
        fake_preds = self.discriminator(fake_imgs)
        adv_loss = self.adv_loss(fake_preds - real_preds.mean(0, keepdims=True), valid)

        # content loss based on the features from a pretrained network
        real_feats = self.feature_extractor(y).detach()
        fake_feats = self.feature_extractor(fake_imgs)
        content_loss = self.content_loss(fake_feats, real_feats)
        
        gen_loss = self.alpha_pixel * pixel_loss + self.alpha_adv * adv_loss + content_loss
        return gen_loss, fake_imgs

    def discriminator_step(self, batch, fake_imgs):
        x, y, _ = batch

        pred_real = self.discriminator(y)
        pred_fake = self.discriminator(fake_imgs.detach())

        valid = torch.ones((x.size(0), *self.discriminator.output_shape), requires_grad=False).type_as(y)
        fake = torch.zeros((x.size(0),*self.discriminator.output_shape), requires_grad=False).type_as(x)

        # relativistic adversarial losses
        real_loss = self.adv_loss(pred_real - pred_fake.mean(0, keepdims=True), valid)
        fake_loss = self.adv_loss(pred_fake - pred_real.mean(0, keepdims=True), fake)

        # discriminator loss is the average of these
        disc_loss = (fake_loss + real_loss) / 2
        return disc_loss

    def training_step(self, train_batch, batch_idx):
        train_batch[0] = self.preprocess(train_batch[0])

        gen_opt, disc_opt = self.optimizers()

        gen_loss, fake_imgs = self.generator_step(train_batch)
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        gen_opt.step()

        self.log('Train/gen_loss', gen_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.global_step >= self._warmup_iters:
            disc_loss = self.discriminator_step(train_batch, fake_imgs)
            disc_opt.zero_grad()
            self.manual_backward(disc_loss)
            disc_opt.step()
            self.log('Train/disc_loss', disc_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, outputs) -> None:
        gen_sch, disc_sch = self.lr_schedulers()
        gen_sch.step()
        disc_sch.step()

    def validation_step(self, val_batch, batch_idx):
        val_batch[0] = self.preprocess(val_batch[0])
        x, y, _ = val_batch

        with torch.no_grad():
            gen_loss, fake_imgs = self.generator_step(val_batch)
            disc_loss = self.discriminator_step(val_batch, fake_imgs)

            m1 = psnr(fake_imgs, y)
            m2 = ssim(fake_imgs, y)

            self.log('Validation/gen_loss', gen_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('Validation/disc_loss', disc_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

            self.log('Validation/psnr', m1.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('Validation/ssim', m2.item(), on_step=True, on_epoch=True, prog_bar=True)
