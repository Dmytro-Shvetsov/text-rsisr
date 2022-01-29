import os
import json
import argparse

from src.datamodule import DataModule
from src.utils.config_reader import Config, object_from_dict
from src.trainers.trainer import Trainer

argparser = argparse.ArgumentParser(description='Script used for training the models.')
argparser.add_argument('--config', '-c', type=str, required=True, help='Configuration file path.')

def seed_everything(seed):
    import random, numpy, torch
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    config = Config(args.config)
    seed_everything(config.seed)
    datamodule = DataModule(config)
    datamodule.setup()
    trainer = Trainer(config)
    trainer.fit(datamodule.train_dataloader(), datamodule.val_dataloader())
    results = trainer.test(datamodule.test_dataloader())
    with open(os.path.join(trainer.model_dir, 'test_results.json'), 'w') as fid:
        json.dump(results, fid)


if __name__ == '__main__':
    train(argparser.parse_args())
