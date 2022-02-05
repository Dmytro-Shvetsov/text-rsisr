from collections import OrderedDict
import datetime
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
        hr_size = visuals['HR'].shape[-2], visuals['HR'].shape[-1]
        for key in visuals:
            if visuals[key].shape != visuals['HR'].shape:
                visuals[key] = F.interpolate(visuals[key], hr_size, mode='bicubic')
        image = torch.cat([make_grid(images, nrow=1, padding=0, normalize=True) for images in visuals.values()], 2)
        cols = '-'.join(visuals.keys())
        save_fp = self.vis_dir / f'{stage}_step_{str(step).zfill(5)}_{cols}.jpg'
        save_image(image, save_fp, normalize=False)

    def plot_scalars(self, losses, step, section, stage=''):
        for loss_name, value in losses.items():
            self.writer.add_scalar(f'{section.title()}/{stage}_{loss_name}', value, step)

    def print_scalars(self, data, epoch, max_epochs, iters, max_iters, t_comp):
        """Console log current losses.

        Parameters:
            data (OrderedDict) -- data to be logged in the format of (name, float) pairs
            epoch (int) -- current epoch
            max_epoch (int) -- total number of epochs
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            max_iters (int) -- total number of iterations in one epoch
            t_comp (float) -- computational time per data point (normalized by batch_size)
        """
        # Determine approximate time left
        batches_done = epoch * max_iters + iters
        batches_left = max_epochs * max_iters - batches_done
        time_left = datetime.timedelta(seconds=batches_left * t_comp)

        message = '(Epoch: %d/%d, iters: %d/%d, time: %.6f, eta: %s) ' % (epoch, max_epochs, iters, max_iters, t_comp, time_left)
        message += ', '.join('%s: %.3f' % item for item in data.items())
        self.logger.info(message.title())

    def reset(self):
        for p in self.log_dir.glob('events*'):
            p.unlink(True)

    def __del__(self):
        self.writer.flush()
