from collections import OrderedDict
import logging

from pathlib import Path
import time
import torch
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image


class Logger:
    def __init__(self, log_dir, batch_size):
        self.batch_size = batch_size
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.vis_dir = self.log_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / 'logs.log'
        logging.basicConfig(
            level=logging.INFO, 
            format= r'[%(asctime)s|%(levelname)s] - %(message)s',
            datefmt=r'%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(self.log_file, 'a')]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"================ Session ({time.strftime('%c')}) ================")

    def save_visuals(self, visuals, step, stage=''):
        if visuals['LR'].shape != visuals['HR'].shape:
            visuals['LR'] = F.interpolate(visuals['LR'], scale_factor=2, mode='bicubic')
        image = torch.cat([make_grid(images, nrow=1, padding=0, normalize=True) for images in visuals.values()], 2)
        cols = '-'.join(visuals.keys())
        save_fp = self.vis_dir / f'{stage}_step_{str(step).zfill(5)}_{cols}.jpg'
        save_image(image, save_fp, normalize=False)

    def plot_scalars(self, losses, step, section, stage=''):
        for loss_name, value in losses.items():
            self.writer.add_scalar(f'{section.title()}/{stage}_{loss_name}', value, step)

    def print_scalars(self, data, epoch, iters, t_comp):
        """Console log current losses.

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
        """
        message = '(Epoch: %d, iters: %d, time: %.6f) ' % (epoch, iters, t_comp)
        message += ', '.join('%s: %.3f' % item for item in data.items())
        self.logger.info(message.title())

    def reset(self):
        for p in self.log_dir.glob('events*'):
            p.unlink(True)

    def __del__(self):
        self.writer.flush()
