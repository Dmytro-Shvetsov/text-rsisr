from io import BytesIO
from bisect import bisect_right

import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class TextZoomDataset(Dataset):
    def __init__(self, lmdb_dir, lr_transforms=None, hr_transforms=None):
        super().__init__()
        self._lmdb_dir = str(lmdb_dir)
        with self.open_lmdb().begin(write=False) as txn:
            self._num_samples = int(txn.get(b'num-samples'))
        # do not persist the connection until the object is moved to a different process
        # (https://github.com/pytorch/vision/issues/689#issuecomment-787215916)
        self._env = None

        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms

        self._images_mode = 'RGB'

    def open_lmdb(self):
        env = lmdb.open(
            self._lmdb_dir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        if not env:
            raise AttributeError(f'Unable to open the lmdb storage from {self._lmdb_dir}')
        return env

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        if self._env is None:
            self._env = self.open_lmdb()

        # indexation is expected to start from 1
        index += 1
        with self._env.begin(write=False) as txn:
            lr_img = self._get_image(txn, b'image_lr-%09d' % index)
            if self.lr_transforms:
                lr_img = self.lr_transforms(image=lr_img)['image']
            lr_img = torch.from_numpy(lr_img).permute(2, 0, 1)

            hr_img = self._get_image(txn, b'image_hr-%09d' % index)
            if self.hr_transforms:
                hr_img = self.hr_transforms(image=hr_img)['image']
            hr_img = torch.from_numpy(hr_img).permute(2, 0, 1)
            text = str(txn.get(b'label-%09d' % index).decode('utf-8'))
        return lr_img, hr_img, text

    def _get_image(self, txn, key):
        imgbuf = txn.get(key)
        if imgbuf is None:
            raise AttributeError(f'Unable to find key {key} in the lmdb storage created from {self._lmdb_dir}')
        buf = BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert(self._images_mode))


def list_cum_sum(list):
    ret, curr_sum = [], 0
    for num in list:
        ret.append(curr_sum + num)
        curr_sum += num
    return ret


class ConcatDataset(Dataset):
    def __init__(self, datasets) -> None:
        self._datasets = datasets
        self._cum_sizes = list_cum_sum(len(d) for d in datasets)

    def __len__(self):
        return self._cum_sizes[-1]

    def __getitem__(self, index):
        dst_idx = bisect_right(self._cum_sizes, index)
        sample_idx = index if dst_idx == 0 else (index - self._cum_sizes[dst_idx - 1])
        return self._datasets[dst_idx][sample_idx]


if __name__ == '__main__':
    root_dir = './data/text_zoom/train1'
    dst = TextZoomDataset(root_dir, (25, 25))
    X, y, _ = dst[1]
