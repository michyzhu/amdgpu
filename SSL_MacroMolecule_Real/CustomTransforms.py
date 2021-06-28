import numpy as np
import torch
import random

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        image_tensor = torch.from_numpy(image.copy())
        return image_tensor

class Random3DRotate(object):
    """Rotate a 3D numpy array of size batchsize * depth * height * width, along x, y and z axis"""

    def __call__(self, image):
        rot = random.randint(0,9)
        if rot == 0: # no rotation
            rotated_img = image
        elif rot == 1:
            rotated_img = np.rot90(image, 1, (2,3))
        elif rot == 2:
            rotated_img = np.rot90(image, 2, (2,3))
        elif rot == 3:
            rotated_img = np.rot90(image, 3, (2,3))
        elif rot == 4:
            rotated_img = np.rot90(image, 1, (1,3))
        elif rot == 5:
            rotated_img = np.rot90(image, 2, (1,3))
        elif rot == 6:
            rotated_img = np.rot90(image, 3, (1,3))
        elif rot == 7:
            rotated_img = np.rot90(image, 1, (1,2))
        elif rot == 8:
            rotated_img = np.rot90(image, 2, (1,2))
        elif rot == 9:
            rotated_img = np.rot90(image, 3, (1,2))

        assert rotated_img.shape == (1,32,32,32)

        return rotated_img
