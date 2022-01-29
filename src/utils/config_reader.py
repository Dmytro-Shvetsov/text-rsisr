import pydoc
from typing import Any
from easydict import EasyDict

import yaml


class Config:
    def __init__(self, file_path):
        params = self.load(file_path)
        self.__dict__.update(params)

    def load(self, file_path):
        with open(file_path, encoding='utf-8') as fid:
            return EasyDict(yaml.load(fid, Loader=yaml.Loader))

    def get(self, param_name, default=None):
        return self.__dict__.get(param_name, default)

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            raise AttributeError(f'Parameter {__name} not found. Available parameters: {list(self.__dict__.keys())}')


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)
