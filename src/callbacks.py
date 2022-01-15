import os
from pathlib import Path
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from PIL import Image

from src.datamodule import denormalize
from src.utils.config_reader import Config


class EpochVisualizationCallback(Callback):

    def __init__(self, cfg:Config, visualizations_dir) -> None:
        super().__init__()
        self.cfg = cfg
        self._freq = cfg.vis_frequency
        self._n_vis_images = cfg.n_vis_images
        self._save_dir = Path(visualizations_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)

    def build_vis_image(self, batch, model) -> None:
        x, y, _ = batch

        x, y = x[:self._n_vis_images], y[:self._n_vis_images]
        y_hat = model(x)
        x = denormalize(x, self.cfg.norm_means, self.cfg.norm_stds, 255.0)
        # Image.fromarray(x.permute(0, 2, 3, 1).byte().detach().numpy()[0]).save('test.jpg')
        y = denormalize(y, self.cfg.norm_means, self.cfg.norm_stds, 255.0)
        y_hat = denormalize(y_hat, self.cfg.norm_means, self.cfg.norm_stds, 255.0)
        images_grid = torch.cat([torch.cat(items, axis=2) for items in zip(y, y_hat)], axis=1)
        # self.logger.experiment.add_image('train/inference_example', images_grid, self.current_epoch)
        return images_grid.permute(1, 2, 0).byte().numpy()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        if batch_idx % self._freq:
            img = self.build_vis_image(batch, pl_module)
            save_fp = os.path.join(self._save_dir, 'train_step_{}_batch_{}.jpg'.format(
                str(trainer.global_step).zfill(len(str(trainer.num_training_batches))),
                str(batch_idx).zfill(len(str(sum(trainer.num_val_batches)))),
            ))
            Image.fromarray(img).save(save_fp)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if batch_idx % self._freq:
            img = self.build_vis_image(batch, pl_module)
            save_fp = os.path.join(self._save_dir, 'validation_step_{}_batch_{}.jpg'.format(
                str(trainer.global_step).zfill(len(str(trainer.num_training_batches))),
                str(batch_idx).zfill(len(str(sum(trainer.num_val_batches)))),
            ))
            Image.fromarray(img).save(save_fp)
