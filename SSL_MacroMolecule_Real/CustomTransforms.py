"""
FileName:	CustomTransforms.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:50:42
"""

import numpy as np
import torch
import random


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        image_tensor = torch.Tensor(image.copy())
        return image_tensor


class Normalize3D(object):
    """Normalize a tensor voxel with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor voxel of size (C, D, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor voxel.
        """
        return normalize3D(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def normalize3D(tensor, mean, std, inplace=False):
    """Normalize a tensor voxel with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    Args:
        tensor (Tensor): Tensor voxel of size (C, D, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor voxel.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 4:
        raise ValueError('Expected tensor to be a tensor voxel of size (C, D, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None, None]
    if std.ndim == 1:
        std = std[:, None, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


class Random3DRotate(object):
    """Rotate a 3D numpy array of size batchsize * depth * height * width, along x, y and z axis"""

    def __call__(self, image):
        rot = random.randint(0, 9)
        if rot == 0:  # no rotation
            rotated_img = image
        elif rot == 1:
            rotated_img = np.rot90(image, 1, (2, 3))
        elif rot == 2:
            rotated_img = np.rot90(image, 2, (2, 3))
        elif rot == 3:
            rotated_img = np.rot90(image, 3, (2, 3))
        elif rot == 4:
            rotated_img = np.rot90(image, 1, (1, 3))
        elif rot == 5:
            rotated_img = np.rot90(image, 2, (1, 3))
        elif rot == 6:
            rotated_img = np.rot90(image, 3, (1, 3))
        elif rot == 7:
            rotated_img = np.rot90(image, 1, (1, 2))
        elif rot == 8:
            rotated_img = np.rot90(image, 2, (1, 2))
        elif rot == 9:
            rotated_img = np.rot90(image, 3, (1, 2))

        assert rotated_img.shape == (1, 32, 32, 32)

        return rotated_img
