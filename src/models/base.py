from abc import abstractmethod, ABC, abstractproperty
from typing import Dict, List, Union

import torch
import numpy as np
import albumentations as albu

from src.utils.config_reader import Config


ArrayLike = Union[np.ndarray, torch.Tensor]


class SuperResolutionModel(torch.nn.Module, ABC):
    cfg:Config
    is_loaded:bool

    def load(self) -> None:
        """
        Restore the model from existing checkpoint.
        """

    @abstractproperty
    def optimizers(self) -> List:
        pass

    @abstractproperty
    def schedulers(self) -> List:
        pass

    def preprocess(self, images:ArrayLike) -> ArrayLike:
        """
        Prepares the images for the model inference.

        Args:
            images (ArrayLike): (N, C, H, W) shaped images array.

        Returns:
            ArrayLike: preprocessed images.
        """
        pass

    @abstractmethod
    def forward(self, inputs:ArrayLike) -> ArrayLike:
        """
        Runs the preprocessed images through the network.

        Args:
            inputs (ArrayLike): preprocessed images.

        Returns:
            ArrayLike: (N, C, s*H, s*W) shaped enhanced images array, where s is the scale factor of the model.
        """
        pass

    @abstractmethod
    def parse_outputs(self, outputs:ArrayLike) -> ArrayLike:
        """
        Performs necessary conversions, castings, etc on the model outputs.

        Args:
            outputs (ArrayLike): raw outputs of the model after inference.

        Returns:
            ArrayLike: (N, C, s*H, s*W) shaped enhanced images array, where s is the scale factor of the model.
        """
        pass

    @abstractmethod
    def training_step(self, batch:ArrayLike) -> Dict:
        """
        Performs a training step to update model's parameters.
        """
        pass

    @abstractmethod
    def eval_step(self, batch:ArrayLike) -> Dict:
        pass

    def process(self, images:ArrayLike) -> ArrayLike:
        ins = self.preprocess(images)
        outs = self(ins)
        return self.parse_outputs(outs)
