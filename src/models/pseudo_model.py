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

from .core.pseudosr.rcan import make_cleaning_net, make_SR_net
from .core.pseudosr.generators import TransferNet
from .core.cyclegan.models import GeneratorResNet, Discriminator
from .core.pseudosr.discriminators import NLayerDiscriminator
from .core.pseudosr.losses import GANLoss, geometry_ensemble

from src.models.esrgan import ESRGAN



from src.losses.gradient_prior import GradientPriorLoss
from .core.pan.models import PAN


class PseudoModel(SuperResolutionModel):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.cfg = config
        self.device = config.device

        self.idt_input_clean = True # corrupted (False) or clean (True)
        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.hr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])
        self.scale_factor = int(config.hr_img_size[0] / config.lr_img_size[0])

        self.G_xy = make_cleaning_net().to(self.device)
        self.G_yx = TransferNet().to(self.device)
        self.U = make_SR_net(scale_factor=self.scale_factor).to(self.device)

        self.D_x = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity, n_group=1).to(self.device)
        self.D_y = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity, n_group=1).to(self.device)
        self.D_sr = NLayerDiscriminator(3, scale_factor=self.scale_factor, norm_layer=nn.Identity, n_group=1).to(self.device)

        self.opt_Gxy = optim.Adam(self.G_xy.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Gyx = optim.Adam(self.G_yx.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dx = optim.Adam(self.D_x.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dy = optim.Adam(self.D_y.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dsr = optim.Adam(self.D_sr.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_U = optim.Adam(self.U.parameters(), lr=config.lr)

        self.lr_Gxy = optim.lr_scheduler.MultiStepLR(self.opt_Gxy, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Gyx = optim.lr_scheduler.MultiStepLR(self.opt_Gyx, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dx = optim.lr_scheduler.MultiStepLR(self.opt_Dx, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dy = optim.lr_scheduler.MultiStepLR(self.opt_Dy, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dsr = optim.lr_scheduler.MultiStepLR(self.opt_Dsr, milestones=config.lr_milestones, gamma=0.5)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=config.lr_milestones, gamma=0.9)

        self.nets = {"G_xy":self.G_xy, "G_yx":self.G_yx, "U":self.U, "D_x":self.D_x, "D_y":self.D_y, "D_sr":self.D_sr}
        self.optims = {"G_xy":self.opt_Gxy, "G_yx":self.opt_Gyx, "U":self.opt_U, "D_x":self.opt_Dx, "D_y":self.opt_Dy, "D_sr":self.opt_Dsr}
        self.lr_decays = {"G_xy":self.lr_Gxy, "G_yx":self.lr_Gyx, "U":self.lr_U, "D_x":self.lr_Dx, "D_y":self.lr_Dy, "D_sr":self.lr_Dsr}
        self.discs = ["D_x", "D_y", "D_sr"]
        self.gens = ["G_xy", "G_yx", "U"]

        self.global_step = 0
        self.gan_loss = GANLoss("lsgan")
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gp_loss = GradientPriorLoss()

        self.d_sr_weight = 0.1
        self.cyc_weight = 1
        self.idt_weight = 2
        self.geo_weight = 1

    def load(self):
        return NotImplemented
    
    @property
    def optimizers(self) -> List:
        return [self.opt_Gxy, self.opt_Gyx, self.opt_Dx, self.opt_Dy, self.opt_Dsr, self.opt_U] if self.training else []

    @property
    def schedulers(self) -> List:     
        return [self.lr_Gxy, self.lr_Gyx, self.lr_Dx, self.lr_Dy, self.lr_Dsr, self.lr_U] if self.training else []

    def net_save(self, folder, shout=False):
        file_name = os.path.join(folder, f"nets_{self.global_step}.pth")
        nets = {k:v.state_dict() for k, v in self.nets.items()}
        optims = {k:v.state_dict() for k, v in self.optims.items()}
        lr_decays = {k:v.state_dict() for k, v in self.lr_decays.items()}
        alls = {"nets":nets, "optims":optims, "lr_decays":lr_decays}
        torch.save(alls, file_name)
        if shout: print("Saved: ", file_name)
        return file_name

    def net_load(self, file_name, strict=True):
        map_loc = {"cuda:0": f"cuda:{self.device}"}
        loaded = torch.load(file_name, map_location=map_loc)
        for n in self.nets:
            self.nets[n].load_state_dict(loaded["nets"][n], strict=strict)
        for o in self.optims:
            self.optims[o].load_state_dict(loaded["optims"][o])
        for l in self.lr_decays:
            self.lr_decays[l].load_state_dict(loaded["lr_decays"][l])

    def net_grad_toggle(self, nets, need_grad):
        for n in nets:
            for p in self.nets[n].parameters():
                p.requires_grad = need_grad

    def mode_selector(self, mode="train"):
        if mode == "train":
            for n in self.nets:
                self.nets[n].train()
        elif mode in ["eval", "test"]:
            for n in self.nets:
                self.nets[n].eval()

    def lr_decay_step(self, shout=False):
        lrs = "\nLearning rates: "
        changed = False
        for i, n in enumerate(self.lr_decays):
            lr_old = self.lr_decays[n].get_last_lr()[0]
            self.lr_decays[n].step()
            lr_new = self.lr_decays[n].get_last_lr()[0]
            if lr_old != lr_new:
                changed = True
                lrs += f", {n}={self.lr_decays[n].get_last_lr()[0]}" if i > 0 else f"{n}={self.lr_decays[n].get_last_lr()[0]}"
        if shout and changed: print(lrs)

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.generator.device)
        if torch.is_floating_point(images): # assumed that already normalized
            return images
        return self._prepr_op(images)
    
    def forward(self, inputs):
        y = self.nets["G_xy"](inputs)
        return self.nets["U"](y)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def training_step(self, batch):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        Xs, Ys  = batch[:2]
        Yds = F.interpolate(Ys, scale_factor=1 / self.scale_factor, mode='bicubic')
        Zs = torch.randn(self.cfg.batch_size, 1, 4, 16, dtype=torch.float32, device=self.device)

        self.global_step += 1
        loss_dict = OrderedDict()

        # forward
        fake_Xs = self.G_yx(Yds, Zs)
        rec_Yds = self.G_xy(fake_Xs)
        fake_Yds = self.G_xy(Xs)
        # geo_Yds = geometry_ensemble(self.G_xy, Xs)
        idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        sr_y = self.U(rec_Yds)
        sr_x = self.U(fake_Yds)

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], True)
        # D_x
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()

        # D_y
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        # D_sr
        # pred_sr_x = self.D_sr(sr_x.detach())
        # pred_sr_y = self.D_sr(sr_y.detach())
        # loss_D_sr = (self.gan_loss(pred_sr_x, True, True) + self.gan_loss(pred_sr_y, False, True)) * 0.5
        # sr_x - real, sr_y - fake
        self.opt_Dsr.zero_grad()
        loss_D_sr, _ = self.U.discriminator_step((Yds, sr_x.detach()), sr_y.detach())
        loss_D_sr.backward()
        self.opt_Dsr.step()
        loss_dict["D_sr"] = loss_D_sr.item()

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], False)
        # G_yx
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        pred_sr_y = self.D_sr(sr_y)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        # loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        loss_d_sr = self.gan_loss(pred_sr_y, True, False)
        loss_geo = 0.0 # temp test
        # loss_d_sr = 0.0 # temp test
        loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_d_sr
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        # loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["D_sr"] = loss_d_sr.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()

        # U

        # self.opt_U.zero_grad()
        # sr_y = self.U(rec_Yds.detach())
        # loss_U_pix = self.l1_loss(sr_y, Ys)
        # loss_U_gp = 1e-4 * self.gp_loss(sr_y, Ys)
        # loss_U = loss_U_pix + loss_U_gp
        self.opt_U.zero_grad()
        loss_U, _, _ = self.U.generator_step((rec_Yds.detach(), Ys)) 
        loss_U.backward()
        self.opt_U.step()
        # loss_dict["U_pix"] = loss_U_pix.item()
        # loss_dict["U_gp"] = loss_U_gp.item()
        loss_dict["U"] = loss_U.item()

        logs = OrderedDict((
            ('losses', loss_dict),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('G_xy', fake_Yds),
                ('G_yx', fake_Xs),
                ('U_G_xy', sr_x),
            ))),
        ))
        return logs

    @torch.no_grad()
    def eval_step(self, batch):
        Xs, Ys = batch[:2]
        Yds = F.interpolate(Ys, scale_factor=1 / self.scale_factor, mode='bicubic')
        Zs = torch.randn(Xs.shape[0], 1, 4, 16, dtype=torch.float32, device=self.device)

        y = self.nets["G_xy"](Xs)
        x = self.nets["G_yx"](Yds, Zs)
        sr = self.nets["U"](y)

        logs = OrderedDict((
            ('losses', OrderedDict()),
            ('metrics', OrderedDict((
                ('G_xy_PSNR', psnr(y, Yds)),
                ('G_xy_SSIM', ssim(y, Yds)),
                ('U_PSNR', psnr(sr, Ys)),
                ('U_SSIM', ssim(sr, Ys)),
            ))),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('G_xy', y),
                ('G_yx', x),
                ('U_G_xy', sr),
            ))),
        ))
        return logs


class PseudoModelESRGAN(SuperResolutionModel):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.cfg = config
        self.device = config.device

        self.idt_input_clean = True # corrupted (False) or clean (True)
        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.hr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])
        self.scale_factor = int(config.hr_img_size[0] / config.lr_img_size[0])

        # self.G_xy = GeneratorResNet((3, *config.lr_img_size), kwargs.get('n_residual_blocks', 9))
        # self.G_yx = GeneratorResNet((3, *config.lr_img_size), kwargs.get('n_residual_blocks', 9))
        self.G_xy = make_cleaning_net().to(self.device)
        self.G_yx = TransferNet().to(self.device)
        # self.U = make_SR_net(scale_factor=self.scale_factor).to(self.device)
        self.U = ESRGAN(config)

        self.D_x = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity, n_group=1).to(self.device)
        self.D_y = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity, n_group=1).to(self.device)
        self.D_sr = self.U.discriminator
        # self.D_sr = NLayerDiscriminator(3, scale_factor=self.scale_factor, norm_layer=nn.Identity, n_group=1).to(self.device)

        self.opt_Gxy = optim.Adam(self.G_xy.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Gyx = optim.Adam(self.G_yx.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dx = optim.Adam(self.D_x.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dy = optim.Adam(self.D_y.parameters(), lr=config.lr, betas=(0.5, 0.999))
        # self.opt_Dsr = optim.Adam(self.D_sr.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dsr = self.U.optimizer_D
        # self.opt_U = optim.Adam(self.U.parameters(), lr=config.lr)
        self.opt_U = self.U.optimizer_G

        self.lr_Gxy = optim.lr_scheduler.MultiStepLR(self.opt_Gxy, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Gyx = optim.lr_scheduler.MultiStepLR(self.opt_Gyx, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dx = optim.lr_scheduler.MultiStepLR(self.opt_Dx, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dy = optim.lr_scheduler.MultiStepLR(self.opt_Dy, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dsr = optim.lr_scheduler.MultiStepLR(self.opt_Dsr, milestones=config.lr_milestones, gamma=0.5)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=config.lr_milestones, gamma=0.9)

        self.nets = {"G_xy":self.G_xy, "G_yx":self.G_yx, "U":self.U, "D_x":self.D_x, "D_y":self.D_y, "D_sr":self.D_sr}
        self.optims = {"G_xy":self.opt_Gxy, "G_yx":self.opt_Gyx, "U":self.opt_U, "D_x":self.opt_Dx, "D_y":self.opt_Dy, "D_sr":self.opt_Dsr}
        self.lr_decays = {"G_xy":self.lr_Gxy, "G_yx":self.lr_Gyx, "U":self.lr_U, "D_x":self.lr_Dx, "D_y":self.lr_Dy, "D_sr":self.lr_Dsr}
        self.discs = ["D_x", "D_y", "D_sr"]
        self.gens = ["G_xy", "G_yx", "U"]

        self.global_step = 0
        self.gan_loss = GANLoss("lsgan")
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gp_loss = GradientPriorLoss()

        self.d_sr_weight = 0.1
        self.cyc_weight = 1
        self.idt_weight = 2
        self.geo_weight = 1

    def load(self):
        return NotImplemented
    
    @property
    def optimizers(self) -> List:
        return [self.opt_Gxy, self.opt_Gyx, self.opt_Dx, self.opt_Dy, self.opt_Dsr, self.opt_U] if self.training else []

    @property
    def schedulers(self) -> List:     
        return [self.lr_Gxy, self.lr_Gyx, self.lr_Dx, self.lr_Dy, self.lr_Dsr, self.lr_U] if self.training else []

    def net_save(self, folder, shout=False):
        file_name = os.path.join(folder, f"nets_{self.global_step}.pth")
        nets = {k:v.state_dict() for k, v in self.nets.items()}
        optims = {k:v.state_dict() for k, v in self.optims.items()}
        lr_decays = {k:v.state_dict() for k, v in self.lr_decays.items()}
        alls = {"nets":nets, "optims":optims, "lr_decays":lr_decays}
        torch.save(alls, file_name)
        if shout: print("Saved: ", file_name)
        return file_name

    def net_load(self, file_name, strict=True):
        map_loc = {"cuda:0": f"cuda:{self.device}"}
        loaded = torch.load(file_name, map_location=map_loc)
        for n in self.nets:
            self.nets[n].load_state_dict(loaded["nets"][n], strict=strict)
        for o in self.optims:
            self.optims[o].load_state_dict(loaded["optims"][o])
        for l in self.lr_decays:
            self.lr_decays[l].load_state_dict(loaded["lr_decays"][l])

    def net_grad_toggle(self, nets, need_grad):
        for n in nets:
            for p in self.nets[n].parameters():
                p.requires_grad = need_grad

    def mode_selector(self, mode="train"):
        if mode == "train":
            for n in self.nets:
                self.nets[n].train()
        elif mode in ["eval", "test"]:
            for n in self.nets:
                self.nets[n].eval()

    def lr_decay_step(self, shout=False):
        lrs = "\nLearning rates: "
        changed = False
        for i, n in enumerate(self.lr_decays):
            lr_old = self.lr_decays[n].get_last_lr()[0]
            self.lr_decays[n].step()
            lr_new = self.lr_decays[n].get_last_lr()[0]
            if lr_old != lr_new:
                changed = True
                lrs += f", {n}={self.lr_decays[n].get_last_lr()[0]}" if i > 0 else f"{n}={self.lr_decays[n].get_last_lr()[0]}"
        if shout and changed: print(lrs)

    def preprocess(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images, device=self.generator.device)
        if torch.is_floating_point(images): # assumed that already normalized
            return images
        return self._prepr_op(images)
    
    def forward(self, inputs):
        y = self.nets["G_xy"](inputs)
        return self.nets["U"](y)

    def parse_outputs(self, outputs):
        return denormalize(outputs, self.means, self.stds).mul_(255).add_(0.5).clamp_(0, 255).byte()

    def training_step(self, batch):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        Xs, Ys  = batch[:2]
        Yds = F.interpolate(Ys, scale_factor=1 / self.scale_factor, mode='bicubic')
        Zs = torch.randn(self.cfg.batch_size, 1, 4, 16, dtype=torch.float32, device=self.device)

        self.global_step += 1
        loss_dict = OrderedDict()

        # forward
        fake_Xs = self.G_yx(Yds, Zs)
        rec_Yds = self.G_xy(fake_Xs)
        fake_Yds = self.G_xy(Xs)
        # geo_Yds = geometry_ensemble(self.G_xy, Xs)
        idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        sr_y = self.U(rec_Yds)
        sr_x = self.U(fake_Yds)

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], True)
        # D_x
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()

        # D_y
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        # D_sr
        pred_sr_x = self.D_sr(sr_x.detach())
        pred_sr_y = self.D_sr(sr_y.detach())
        self.opt_Dsr.zero_grad()
        # sr_x - real, sr_y - fake
        loss_D_sr = (self.gan_loss(pred_sr_x, True, True) + self.gan_loss(pred_sr_y, False, True)) * 0.5
        loss_D_sr.backward()
        self.opt_Dsr.step()
        loss_dict["D_sr"] = loss_D_sr.item()

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], False)
        # G_yx
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        pred_sr_y = self.D_sr(sr_y)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        # loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        loss_d_sr = self.gan_loss(pred_sr_y, True, False)
        loss_geo = 0.0 # temp test
        # loss_d_sr = 0.0 # temp test
        loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_d_sr
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        # loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["D_sr"] = loss_d_sr.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()

        # U
        self.opt_U.zero_grad()
        sr_y = self.U(rec_Yds.detach())
        loss_U_pix = self.l1_loss(sr_y, Ys)
        loss_U_gp = 1e-4 * self.gp_loss(sr_y, Ys)
        loss_U = loss_U_pix + loss_U_gp
        loss_U.backward()
        self.opt_U.step()
        loss_dict["U_pix"] = loss_U_pix.item()
        loss_dict["U_gp"] = loss_U_gp.item()
        loss_dict["U"] = loss_U.item()

        logs = OrderedDict((
            ('losses', loss_dict),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('G_xy', fake_Yds),
                ('G_yx', fake_Xs),
                ('U_G_xy', sr_x),
            ))),
        ))
        return logs

    @torch.no_grad()
    def eval_step(self, batch):
        Xs, Ys = batch[:2]
        Yds = F.interpolate(Ys, scale_factor=1 / self.scale_factor, mode='bicubic')
        Zs = torch.randn(Xs.shape[0], 1, 4, 16, dtype=torch.float32, device=self.device)

        y = self.nets["G_xy"](Xs)
        x = self.nets["G_yx"](Yds, Zs)
        sr = self.nets["U"](y)

        logs = OrderedDict((
            ('losses', OrderedDict()),
            ('metrics', OrderedDict((
                ('G_xy_PSNR', psnr(y, Yds)),
                ('G_xy_SSIM', ssim(y, Yds)),
                ('U_PSNR', psnr(sr, Ys)),
                ('U_SSIM', ssim(sr, Ys)),
            ))),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('G_xy', y),
                ('G_yx', x),
                ('U_G_xy', sr),
            ))),
        ))
        return logs


class PseudoModelPAN(PseudoModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.cfg = config
        self.device = config.device

        self.idt_input_clean = True # corrupted (False) or clean (True)
        self.means, self.stds = torch.Tensor(config.norm_means).to(self.cfg.device), torch.Tensor(config.norm_stds).to(self.cfg.device)
        self._prepr_op = Compose([
            Resize(self.cfg.hr_img_size, InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(self.means, self.stds),
        ])
        self.scale_factor = int(config.hr_img_size[0] / config.lr_img_size[0])

        # self.G_xy = GeneratorResNet((3, *config.lr_img_size), kwargs.get('n_residual_blocks', 9))
        # self.G_yx = GeneratorResNet((3, *config.lr_img_size), kwargs.get('n_residual_blocks', 9))
        self.G_xy = make_cleaning_net().to(self.device)
        self.G_yx = TransferNet().to(self.device)
        self.U = PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=self.scale_factor)

        self.D_x = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity, n_group=1).to(self.device)
        self.D_y = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity, n_group=1).to(self.device)
        self.D_sr = NLayerDiscriminator(3, scale_factor=self.scale_factor, norm_layer=nn.Identity, n_group=1).to(self.device)

        self.opt_Gxy = optim.Adam(self.G_xy.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Gyx = optim.Adam(self.G_yx.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dx = optim.Adam(self.D_x.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dy = optim.Adam(self.D_y.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_Dsr = optim.Adam(self.D_sr.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_U = optim.Adam(self.U.parameters(), lr=config.lr)

        self.lr_Gxy = optim.lr_scheduler.MultiStepLR(self.opt_Gxy, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Gyx = optim.lr_scheduler.MultiStepLR(self.opt_Gyx, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dx = optim.lr_scheduler.MultiStepLR(self.opt_Dx, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dy = optim.lr_scheduler.MultiStepLR(self.opt_Dy, milestones=config.lr_milestones, gamma=0.5)
        self.lr_Dsr = optim.lr_scheduler.MultiStepLR(self.opt_Dsr, milestones=config.lr_milestones, gamma=0.5)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=config.lr_milestones, gamma=0.9)

        self.nets = {"G_xy":self.G_xy, "G_yx":self.G_yx, "U":self.U, "D_x":self.D_x, "D_y":self.D_y, "D_sr":self.D_sr}
        self.optims = {"G_xy":self.opt_Gxy, "G_yx":self.opt_Gyx, "U":self.opt_U, "D_x":self.opt_Dx, "D_y":self.opt_Dy, "D_sr":self.opt_Dsr}
        self.lr_decays = {"G_xy":self.lr_Gxy, "G_yx":self.lr_Gyx, "U":self.lr_U, "D_x":self.lr_Dx, "D_y":self.lr_Dy, "D_sr":self.lr_Dsr}
        self.discs = ["D_x", "D_y", "D_sr"]
        self.gens = ["G_xy", "G_yx", "U"]

        self.global_step = 0
        self.gan_loss = GANLoss("lsgan")
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gp_loss = GradientPriorLoss()

        self.d_sr_weight = 0.1
        self.cyc_weight = 1
        self.idt_weight = 2
        self.geo_weight = 1

    def training_step(self, batch):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        Xs, Ys  = batch[:2]
        Yds = F.interpolate(Ys, scale_factor=1 / self.scale_factor, mode='bicubic')
        Zs = torch.randn(self.cfg.batch_size, 1, 4, 16, dtype=torch.float32, device=self.device)

        self.global_step += 1
        loss_dict = OrderedDict()

        # forward
        fake_Xs = self.G_yx(Yds, Zs)
        rec_Yds = self.G_xy(fake_Xs)
        fake_Yds = self.G_xy(Xs)
        # geo_Yds = geometry_ensemble(self.G_xy, Xs)
        idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        sr_y = self.U(rec_Yds)
        sr_x = self.U(fake_Yds)

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], True)
        # D_x
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()

        # D_y
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        # D_sr
        pred_sr_x = self.D_sr(sr_x.detach())
        pred_sr_y = self.D_sr(sr_y.detach())
        # sr_x - real, sr_y - fake
        loss_D_sr = (self.gan_loss(pred_sr_x, True, True) + self.gan_loss(pred_sr_y, False, True)) * 0.5
        loss_dict["D_sr"] = loss_D_sr.item()

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], False)
        # G_yx
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        pred_sr_y = self.D_sr(sr_y)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        # loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        loss_d_sr = self.gan_loss(pred_sr_y, True, False)
        loss_geo = 0.0 # temp test
        loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_d_sr
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        # loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["D_sr"] = loss_d_sr.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()

        # U
        self.opt_U.zero_grad()
        sr_y = self.U(rec_Yds.detach())
        loss_U_pix = self.l1_loss(sr_y, Ys)
        loss_U_gp = 1e-4 * self.gp_loss(sr_y, Ys)
        loss_U = loss_U_pix + loss_U_gp
        loss_dict["U_pix"] = loss_U_pix.item()
        loss_dict["U_gp"] = loss_U_gp.item()
        loss_dict["U"] = loss_U.item()

        logs = OrderedDict((
            ('losses', loss_dict),
            ('images', OrderedDict((
                ('LR', Xs),
                ('HR', Ys),
                ('G_xy', fake_Yds),
                ('G_yx', fake_Xs),
                ('U_G_xy', sr_x),
            ))),
        ))
        return logs
