from typing import OrderedDict
from copy import deepcopy


class AverageMeter:
    def __init__(self) -> None:
        self._data = None

    def reset(self) -> None:
        if not self._data:
            return
        self._data.clear()

    def update(self, new_data:OrderedDict[str, float]) -> OrderedDict[str, float]:
        if self._data is None:
            self._data = new_data
            return new_data
        for k in self._data:
            self._data[k] = (self._data[k] + new_data[k]) / 2
        return self._data
