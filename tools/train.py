import os
import json
import argparse

import torch

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
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.cuda.manual_seed(seed)


def dump_test_results(results, model_dir):
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu().item()
        results[k] = round(v, 5)
    fp = os.path.join(model_dir, 'test_results.json')
    with open(fp, 'w') as fid:
        json.dump(results, fid, indent=4)
    print(f'Saved test results at: {repr(fp)}')


def train(args):
    config = Config(args.config)
    seed_everything(config.seed)
    datamodule = DataModule(config)
    datamodule.setup()
    trainer = Trainer(config)
    trainer.restore_checkpoint()
    trainer.fit(datamodule.train_dataloader(), datamodule.val_dataloader())
    datamodule.setup('test')
    results = trainer.test(datamodule.test_dataloader())
    results['epoch'] = trainer.current_epoch
    dump_test_results(results, trainer.model_dir)


if __name__ == '__main__':
    train(argparser.parse_args())
