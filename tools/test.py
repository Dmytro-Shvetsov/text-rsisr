import os
import json
import argparse

import torch
from PIL import Image
import pytesseract as tesseract
import albumentations as albu
import cv2
from tqdm import tqdm

from src.datamodule import DataModule
from src.utils.average_meter import AverageMeter
from src.utils.config_reader import Config, object_from_dict
from src.trainers.trainer import Trainer
from src.metrics import relative_distance

argparser = argparse.ArgumentParser(description='Script used for training the models.')
argparser.add_argument('--config', '-c', type=str, required=True, help='Configuration file path.')


def train(args):
    config = Config(args.config)
    datamodule = DataModule(config)
    datamodule.setup()
    trainer = Trainer(config)
    trainer.restore_checkpoint()

    # specify path to the preinstalled binary file of tesseract
    tesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print('Tesseract version: ', tesseract.get_tesseract_version())
    
    model = trainer.model
    acc = AverageMeter()
    for batch in tqdm(datamodule.val_dataloader()):
        x, y, true_text = batch
        sr = model.process(x.to(trainer.device)).permute(0, 2, 3, 1)
        
        x = model.parse_outputs(x.to(trainer.device)).permute(0, 2, 3, 1)
        y = model.parse_outputs(y.to(trainer.device)).permute(0, 2, 3, 1)

        batch_accs = {}
        # print(sr.shape, sr.dtype)
        # Image.fromarray(sr.cpu().numpy()[0]).save('test.png')
        # Image.fromarray(sr.cpu().numpy()[1]).save('test2.png')
        pred_text = [tesseract.image_to_string(im).strip() for im in x.cpu().numpy()]
        # print(pred_text, true_text)
        batch_results = [relative_distance(true, pred) for true, pred in zip(true_text, pred_text) if true]
        batch_accs['val_ocr_lr_acc'] = sum(batch_results) / len(batch_results)

        pred_text = [tesseract.image_to_string(im).strip() for im in y.cpu().numpy()]
        # print(pred_text, true_text)
        batch_results = [relative_distance(true, pred) for true, pred in zip(true_text, pred_text) if true]
        batch_accs['val_ocr_hr_acc'] = sum(batch_results) / len(batch_results)

        pred_text = [tesseract.image_to_string(im).strip() for im in sr.cpu().numpy()]
        # print(pred_text, true_text)
        batch_results = [relative_distance(true, pred) for true, pred in zip(true_text, pred_text) if true]
        batch_accs['val_ocr_sr_acc'] = sum(batch_results) / len(batch_results)

        acc.update(batch_accs)

    print('Validation results:', acc.get_average())
    datamodule.setup('test')
    acc = AverageMeter()
    for batch in tqdm(datamodule.test_dataloader()):
        x, y, true_text = batch
        sr = model.process(x.to(trainer.device)).permute(0, 2, 3, 1)
        
        x = model.parse_outputs(x.to(trainer.device)).permute(0, 2, 3, 1)
        y = model.parse_outputs(y.to(trainer.device)).permute(0, 2, 3, 1)

        batch_accs = {}
        pred_text = [tesseract.image_to_string(im).strip() for im in x.cpu().numpy()]
        batch_results = [relative_distance(true, pred) for true, pred in zip(true_text, pred_text) if true]
        batch_accs['test_ocr_lr_acc'] = sum(batch_results) / len(batch_results)

        pred_text = [tesseract.image_to_string(im).strip() for im in y.cpu().numpy()]
        batch_results = [relative_distance(true, pred) for true, pred in zip(true_text, pred_text) if true]
        batch_accs['test_ocr_hr_acc'] = sum(batch_results) / len(batch_results)

        pred_text = [tesseract.image_to_string(im).strip() for im in sr.cpu().numpy()]
        batch_results = [relative_distance(true, pred) for true, pred in zip(true_text, pred_text) if true]
        batch_accs['test_ocr_sr_acc'] = sum(batch_results) / len(batch_results)

        acc.update(batch_accs)

    print('Test results:', acc.get_average())


if __name__ == '__main__':
    train(argparser.parse_args())
