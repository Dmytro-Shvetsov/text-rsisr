import os
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from PIL import Image

from src.datamodule import denormalize
from src.utils.config_reader import Config
from torchvision.utils import save_image, make_grid

class EpochVisualizationCallback(Callback):

    def __init__(self, cfg:Config, visualizations_dir) -> None:
        super().__init__()
        self.cfg = cfg
        self._freq = cfg.vis_frequency
        self._n_vis_images = cfg.n_vis_images
        self._save_dir = Path(visualizations_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self.reset()

    def reset(self):
        for p in self._save_dir.glob('*.jpg'):
            p.unlink(True)

    def build_vis_image(self, batch, model) -> None:
        x, y, _ = batch

        x, y = x[:self._n_vis_images], y[:self._n_vis_images]
        y_hat = model(x)

        x = F.interpolate(x, scale_factor=self.cfg.scale_factor)

        images_grid = torch.cat((x, y, y_hat), axis=-1)
        images_grid = denormalize(images_grid, self.cfg.norm_means, self.cfg.norm_stds)
        images_grid = make_grid(images_grid, nrow=1)
        
        # save_image(images_grid, 'test3.jpg')
        images_grid.mul_(255).add_(0.5).clamp_(0, 255)
        return images_grid.permute(1, 2, 0).byte().cpu().numpy()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        if trainer.global_step % self._freq == 0:
            img = self.build_vis_image(batch, pl_module)
            save_fp = os.path.join(self._save_dir, 'train_step_{}_batch_{}.jpg'.format(
                str(trainer.global_step).zfill(len(str(trainer.num_training_batches))),
                str(batch_idx).zfill(len(str(sum(trainer.num_val_batches)))),
            ))
            Image.fromarray(img).save(save_fp)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if batch_idx % self._freq == 0:
            img = self.build_vis_image(batch, pl_module)
            save_fp = os.path.join(self._save_dir, 'validation_step_{}_batch_{}.jpg'.format(
                str(trainer.global_step).zfill(len(str(trainer.num_training_batches))),
                str(batch_idx).zfill(len(str(sum(trainer.num_val_batches)))),
            ))
            Image.fromarray(img).save(save_fp)
