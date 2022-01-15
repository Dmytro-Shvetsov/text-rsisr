from abc import abstractmethod, ABC
from typing import Union

import torch
import numpy as np
import albumentations as albu

from src.utils.config_reader import Config


ArrayLike = Union[np.ndarray, torch.Tensor]


class SuperResolutionModelInterface(ABC):
    cfg:Config
    _is_loaded:bool

    def load(self) -> None:
        """
        Restore the model from existing checkpoint.
        """

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

    def process(self, images:ArrayLike) -> ArrayLike:
        ins = self.preprocess(images)
        outs = self(ins)
        return self.parse_outputs(outs)
