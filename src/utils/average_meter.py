from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
from copy import deepcopy


class AverageMeter:
    def __init__(self) -> None:
        self._data = None
        self._num_updates = 0

    def reset(self) -> None:
        if not self._data:
            return
        self._data.clear()
        self._num_updates = 0

    def update(self, new_data:OrderedDictType[str, float]) -> None:
        self._num_updates += 1
        if self._data is None:
            self._data = new_data
            return new_data
        for k in self._data:
            self._data[k] += new_data[k]

    def get_average(self):
        return OrderedDict((k, v / self._num_updates) for k, v in self._data.items())
