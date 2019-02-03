###
# This script is extended torchvision.transforms for data which has 
# more than 3 dimensions or wider range.
# And also able to set image on target to image segmentation tasks.
###

import collections
import cv2
import numbers
import numpy as np
import skimage
from scipy import ndimage
import types
import torch
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sources, targets=None):
        if not isinstance(sources, tuple):
            sources = (sources, )
        if targets is None:
            for transform in self.transforms:
                sources = transform(sources)
            return [*sources]
        else:
            if not isinstance(targets, tuple):
                targets = (targets, )
            for transform in self.transforms:
                sources, targets = transform(sources, targets)
            return [*sources, *targets]

    def __repr__(self):
        rep = [ str(t) for t in self.transforms ]
        rep = '\n'.join(rep)
        return rep

class ToTensor(object):
    def __init__(self):
        pass

    def _to_tensor(self, array, dtype):
        if dtype == 'float':
            return torch.from_numpy(array.copy()).float()
        if dtype == 'long':
            return torch.from_numpy(array.copy()).squeeze().long()

    def __call__(self, sources, targets=None):
        if targets is None:
            sources = [ self._to_tensor(x, 'float') for x in sources ]
            return sources
        else:
            sources = [ self._to_tensor(x, 'float') for x in sources ]
            targets = [ self._to_tensor(x, 'long') for x in targets ]
            return sources, targets

class Normalize(object):
    def __init__(self, means, stds):
        if not isinstance(means, tuple):
            means = (means, )
        if not isinstance(stds, tuple):
            stds = (stds, )
        self.means = means
        self.stds = stds

    def _normalize(self, array, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        if not array.shape[0] == mean.shape[0] == std.shape[0]:
            raise TypeError('Array and mean and std are not the same dimension.')

        if mean.ndim == 1:
            mean = mean.reshape(mean.size, 1, 1)
        if std.ndim == 1:
            std = std.reshape(std.size, 1, 1)
        return (array - mean) / np.clip(std, 1e-10, 1e+10)

    def __call__(self, sources, targets=None):
        if not len(sources) == len(self.means) == len(self.stds):
            raise TypeError('source and means is not same length.')

        sources = [ self._normalize(x, m, s) for (x, m, s) in zip(sources, self.means, self.stds) ]
        if targets is None:
            return sources
        else:
            return sources, targets

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def _center_crop(self, array, size):
        if array.ndim == 2:
            h, w = array.shape
            upper = int((h - size) / 2)
            left = int((w - size) / 2)
            return array[upper:upper+size, left:left+size]
        elif array.ndim == 3:
            h, w = array.shape[1:]
            upper = int((h - size) / 2)
            left = int((w - size) / 2)
            return array[:, upper:upper+size, left:left+size]
        else:
            raise TypeError('array.ndim should be 2 or 3, but got {0}'.format(array.ndim))

    def __call__(self, sources, targets=None):
        if targets is None:
            sources = [ self._center_crop(x, self.size) for x in sources ]
            return sources
        else:
            sources = [ self._center_crop(x, self.size) for x in sources ]
            targets = [ self._center_crop(x, self.size) for x in targets ]
            return sources, targets

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def _crop(self, array, upper, left, hsize, wsize):
        if array.ndim == 2:
            return array[upper:upper+hsize, left:left+wsize]
        elif array.ndim == 3:
            return array[:, upper:upper+hsize, left:left+wsize]
        else:
            raise TypeError('array.ndim should be 2 or 3, but got {0}'.format(array.ndim))

    @staticmethod
    def get_params(array, size):
        h = array.shape[-2]
        w = array.shape[-1]
        hsize, wsize = size
        if (w == wsize) and (h == hsize):
            return 0, 0, h, w
        else:
            upper = np.random.randint(0, h - hsize + 1)
            left = np.random.randint(0, w - wsize + 1)
            return upper, left, hsize, wsize

    def __call__(self, sources, targets=None):
        i, j, h, w = self.get_params(sources[0], self.size)
        if targets is None:
            sources = [ self._crop(x, i, j, h, w) for x in sources ]
            return sources
        else:
            sources = [ self._crop(x, i, j, h, w) for x in sources ]
            targets = [ self._crop(x, i, j, h, w) for x in targets ]
            return sources, targets

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = min(abs(p), 1.)

    def __call__(self, sources, targets=None):
        if np.random.random() < self.p:
            if targets is None:
                sources = [ np.flip(x, axis=-1) for x in sources ]
                return sources
            else:
                sources = [ np.flip(x, axis=-1) for x in sources ]
                targets = [ np.flip(x, axis=-1) for x in targets ]
                return sources, targets
        else:
            if targets is None:
                return sources
            else:
                return sources, targets

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = min(abs(p), 1.0)

    def __call__(self, sources, targets=None):
        if np.random.random() < self.p:
            if targets is None:
                sources = [ np.flip(x, axis=-2) for x in sources ]
                return sources, targets
            else:
                sources = [ np.flip(x, axis=-2) for x in sources ]
                targets = [ np.flip(x, axis=-2) for x in targets ]
                return sources, targets
        else:
            if targets is None:
                return sources
            else:
                return sources, targets

class Resize(object):
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def _resize(self, array, size, mode):
        if mode == 'nearest':
            order = 0
        elif mode == 'bilinear':
            order = 1
        elif mode == 'cubic':
            order = 3
        else:
            raise TypeError('mode should be nearest or bilinear or cubic, but got {0}'.format(mode))

        mag = size / array.shape[-1]
        if array.ndim == 2:
            return ndimage.zoom(array, zoom=(mag, mag), order=order)
        elif array.ndim == 3:
            return ndimage.zoom(array, zoom=(1., mag, mag), order=order)
        else:
            raise TypeError('array.ndim should be 2 or 3, but got {0}'.format(array.ndim))

    def __call__(self, sources, targets=None):
        if targets is None:
            sources = [ self._resize(x, self.size, self.mode) for x in sources ]
            return sources
        else:
            sources = [ self._resize(x, self.size, self.mode) for x in sources ]
            targets = [ self._resize(x, self.size, self.mode) for x in targets ]
            return sources, targets

class RandomRotation(object):
    def __init__(self, degrees, reshape=False):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number, it must be positive.')
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence, it must be of len 2.')
            self.degrees = degrees
        self.reshape = reshape

    def _center_crop(self, array, size):
        if array.ndim == 2:
            h, w = array.shape
            upper = int((h - size) / 2)
            left = int((w - size) / 2)
            return array[upper:upper+size, left:left+size]
        elif array.ndim == 3:
            h, w = array.shape[1:]
            upper = int((h - size) / 2)
            left = int((w - size) / 2)
            return array[:, upper:upper+size, left:left+size]

    def _rotate(self, array, angle):
        if not ((array.ndim == 2) or (array.ndim == 3)):
            raise TypeError('array.ndim should be 2 or 3, but got {0}'.format(array.ndim))
        original_size = array.shape[-1]
        array = ndimage.interpolation.rotate(array, angle, axes=(-2, -1), reshape=self.reshape)
        if self.reshape:
            array = self._center_crop(array, original_size)
        return array

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sources, targets=None):
        size = sources[0].shape[-1]
        self.centercrop = CenterCrop(size)
        angle = self.get_params(self.degrees)
        if targets is None:
            sources = [ self._rotate(x, angle) for x in sources ]
            return sources
        else:
            sources = [ self._rotate(x, angle) for x in sources ]
            targets = [ self._rotate(x, angle) for x in targets ]
            return sources, targets

class RandomRotate90(object):
    def __init__(self):
        pass

    def _rotate90(self, array, deg):
        if not ((array.ndim == 2) or (array.ndim == 3)):
            raise TypeError('array.ndim should be 2 or 3, but got {0}'.format(array.ndim))
        array = np.rot90(array, k=deg, axes=(-2, -1))
        return array

    @staticmethod
    def get_params():
        deg = np.random.randint(4)
        return deg

    def __call__(self, sources, targets=None):
        size = sources[0].shape[-1]
        deg = self.get_params()
        if targets is None:
            sources = [ self._rotate90(x, deg) for x in sources ]
            return sources
        else:
            sources = [ self._rotate90(x, deg) for x in sources ]
            targets = [ self._rotate90(x, deg) for x in targets ]
            return sources, targets

class RandomNoise(object):
    """TODO
    if dtype is uint8, var should be 0.01 is best
    if dtype is uint16, var should be 0.001 is best
    """
    def __init__(self, dtype='uint8', mode='gaussian', var=0.01):
        if dtype not in ('uint8', 'uint16'):
            raise NotImplementedError('Support only uint8 or uint16.')
        self.dtype = dtype
        self.mode = mode
        self.var = var

    def _random_noise(self, array, dtype='uint8', mode='gaussian', mean=0., var=0.01):
        if dtype == 'uint8':
            array = array.astype(np.uint8)
        if dtype == 'uint16':
            array = array.astype(np.uint16)

        array = skimage.util.random_noise(array, mode, mean=mean, var=var)
        if dtype == 'uint8':
            array = (array * np.iinfo(np.uint8).max).astype(np.uint8)
        if dtype == 'uint16':
            array = (array * np.iinfo(np.uint32).max).astype(np.uint32)
        return array

    def __call__(self, sources, targets=None):
        sources = [ self._random_noise(x, self.dtype, self.mode, self.var) for x in sources ]
        if targets is None:
            return sources
        else:
            return sources, targets

class RandomBrightnessChange(object):
    def __init__(self, factor=0.5):
        self.factor = factor

    def _random_brightness_change(self, array, factor):
        if not (array.dtype == 'uint8') and (array.dim == 3):
            raise NotImplementedError('Support only 3band, 8bit.')
        hsv = cv2.cvtColor(array.transpose(1,2,0), cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.transpose(2,0,1)

    def __call__(self, sources, targets=None):
        sources = [ self._random_brightness_change(x, self.factor) for x in sources ]
        if targets is None:
            return sources
        else:
            return sources, targets

class ElasticTransform(object):
    def __init__(self, alpha=1000, sigma=30, spline_order=1, mode='nearest'):
        self.alpha = alpha
        self.sigma = sigma
        self.spline_order = spline_order
        self.mode = mode

    def _elastic_transform(self, array, alpha, sigma, spline_order, mode):
        """
        Elastic deformation of image as described in [Simard2003]_.
        ..[Simard2003] Simard, Steinkraus and Platt, "Best Practice for
        Convolutional Neural Networks applied to Visual Document Analysis",
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """

        if array.ndim !=3:
            raise NotImplementedError('Support only 3-dimension array.')
        shape = array.shape[1:]

        dx = ndimage.filters.gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                              sigma, mode='constant', cval=0) * alpha
        dy = ndimage.filters.gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                              sigma, mode='constant', cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        output = np.empty_like(array)
        for i in range(array.shape[0]):
            output[i, :, :] = ndimage.interpolation.map_coordinates(
                array[i, :, :], indices, order=spline_order, mode=mode).reshape(shape)
            return output

    def __call__(self, sources, targets=None):
        sources = [ self._elastic_transform(x, self.alpha, self.sigma, self.spline_order, self.mode) 
                    for x in sources ]
        if targets is None:
            return sources
        else:
            return sources, targets
