from time import perf_counter
from pathlib import Path

import torch

from src import models
from src.models.base import SuperResolutionModel
from src.utils.config_reader import Config, object_from_dict
from src.logger import Logger


class Trainer:
    def __init__(self, config:Config) -> None:
        super().__init__()
        self.config = config

        self.device = self.config.device

        self.model:SuperResolutionModel = object_from_dict(self.config.model, parent=models, config=config)

        # for p in self.model.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        self.model_dir = Path(self.config.logs_dir) / self.config.run_name
        self.ckpt_dir = self.model_dir / 'checkpoints'
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)

        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        self.global_step = 0
        self.current_epoch = 0

        self.vis = Logger(self.model_dir, self.batch_size)
        self.logger = self.vis.logger

        self.save_interval = self.config.get('save_interval', 1)
        self.sample_interval = self.config.get('sample_interval', 1)

    def training_epoch(self, train_loader):
        self.model.train()
        for i, batch in enumerate(train_loader):
            batch[:2] = [t.to(self.device) for t in batch[:2]]
            start_time = perf_counter()
            outs = self.model.training_step(batch)
            time_taken = perf_counter() - start_time
            
            losses, images = outs['losses'], outs['images']
            self.vis.print_scalars(losses, self.current_epoch, self.num_epochs, i, len(train_loader), time_taken)
            if self.global_step % self.sample_interval == 0:
                self.vis.plot_scalars(losses, self.global_step, 'Loss', 'train')
                self.vis.save_visuals(images, self.global_step, 'train')
            self.global_step += 1

    def validation_epoch(self, val_loader):
        self.model.eval()
        for i, batch in enumerate(val_loader):
            batch[:2] = [t.to(self.device) for t in batch[:2]]
            start_time = perf_counter()
            outs = self.model.eval_step(batch)
            time_taken = perf_counter() - start_time

            losses, metrics, images = outs['losses'], outs['metrics'], outs['images']
            self.vis.print_scalars(metrics, self.current_epoch, self.num_epochs, i, len(val_loader), time_taken)
            if i % self.sample_interval == 0:
                test_batches_done = self.current_epoch * len(val_loader) + i
                self.vis.plot_scalars(losses, test_batches_done, 'Loss', 'validation')
                self.vis.plot_scalars(metrics, test_batches_done, 'Metric', 'validation')
                self.vis.save_visuals(images, test_batches_done, 'validation')

    def test_epoch(self, test_loader):
        self.model.eval()
        for i, batch in enumerate(test_loader):
            batch[:2] = [t.to(self.device) for t in batch[:2]]
            start_time = perf_counter()
            outs = self.model.eval_step(batch)
            time_taken = perf_counter() - start_time

            losses, metrics, images = outs['losses'], outs['metrics'], outs['images']
            self.vis.print_scalars(metrics, self.current_epoch, self.num_epochs, i, len(test_loader), time_taken)
            if i % self.sample_interval == 0:
                test_batches_done = self.current_epoch * len(test_loader) + i
                self.vis.plot_scalars(losses, test_batches_done, 'Loss', 'test')
                self.vis.plot_scalars(metrics, test_batches_done, 'Metric', 'test')
                self.vis.save_visuals(images, test_batches_done, 'test')

    def store_checkpoint(self):
        ckpt = {
            'state_dict': self.model.state_dict(),
            'current_epoch': self.current_epoch,
        }
        torch.save(ckpt, self.ckpt_dir / f'epoch-{self.current_epoch}.pth')

    def restore_checkpoint(self):
        ckpt_path = self.config.get('ckpt_path')
        if ckpt_path:
            ckpt = torch.load(ckpt_path, self.device)
            self.current_epoch = ckpt['current_epoch']
            self.global_step = self.batch_size * self.current_epoch
            self.model.global_step = self.global_step
            self.model.load_state_dict(ckpt['state_dict'])
            self.logger.info(f'Successfully restored checkpoint {repr(ckpt_path)}. Continuing epoch {self.current_epoch}...')
        self.model.to(self.device)

    def fit(self, train_loader, val_loader, test_loader=None):
        self.restore_checkpoint()
        self.vis.reset()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.training_epoch(train_loader)
            self.validation_epoch(val_loader)
            for sch in self.model.schedulers:
                sch.step()
            if epoch % self.save_interval == 0:
                self.store_checkpoint()

    def test(self, test_loader):
        # TODO: add average meter for scalar results
        return self.test_epoch(test_loader)
