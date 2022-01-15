import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src import models
from src.utils.config_reader import Config, object_from_dict
from src.datamodule import LitDataModule
from src.callbacks import EpochVisualizationCallback

argparser = argparse.ArgumentParser(description='Script used for training the models.')
argparser.add_argument('--config', '-c', type=str, help='Configuration file path.')


def train(args):
    config = Config(args.config)
    pl.seed_everything(config.seed, workers=True)

    data_module = LitDataModule(config)
    model:pl.LightningDataModule = object_from_dict(config.model, parent=models, config=config)
    logger = TensorBoardLogger(config.logs_dir, default_hp_metric=False)
    callbacks = [
        LearningRateMonitor(log_momentum=True), 
        ModelCheckpoint('saved_models', save_top_k=3, save_last=True, monitor='Validation/gen_loss'),
        EpochVisualizationCallback(config, os.path.join(logger.log_dir, 'visualizations'))
    ]

    trainer:pl.Trainer = object_from_dict(config.trainer, logger=logger, deterministic=True, callbacks=callbacks)
    trainer.fit(model, data_module, ckpt_path=config.ckpt_path)
    # trainer.test(model, data_module)


if __name__ == '__main__':
    train(argparser.parse_args())
