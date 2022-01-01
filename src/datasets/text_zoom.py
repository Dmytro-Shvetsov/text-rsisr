from io import BytesIO

import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class TextZoomDataset(Dataset):
    def __init__(self, lmdb_dir):
        super().__init__()
        self._env = lmdb.open(lmdb_dir)
        self._txn = self._env.begin()
        self._num_samples = int(self._txn.get(b'num-samples'))
        self._images_mode = 'RGB'

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        return {
            'HR_image': torch.from_numpy(self._get_image(b'image_hr-%09d' % index)),
            'LR_image': torch.from_numpy(self._get_image( b'image_lr-%09d' % index)),
            'text': str(self._txn.get(b'label-%09d' % index).decode('utf-8'))
        }

    def _get_image(self, key):
        imgbuf = self._txn.get(key)
        buf = BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert(self._images_mode))


if __name__ == '__main__':
    root_dir = './data/train1'
    dst = TextZoomDataset(root_dir)
    print(len(dst))
    root_dir = './data/train2'
    dst = TextZoomDataset(root_dir)
    print(len(dst))
