import numbers
import random
import cv2
import numpy as np


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Implementation from WLASL repository:
    https://github.com/dxli94/WLASL/blob/master/code/I3D/videotransforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):

        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i + h, j:j + w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Implementation from WLASL repository:
    https://github.com/dxli94/WLASL/blob/master/code/I3D/videotransforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i + th, j:j + tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Implementation from WLASL repository:
    https://github.com/dxli94/WLASL/blob/master/code/I3D/videotransforms.py

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomBC(object):
    """Apply random brightness and contrast
    """

    def __init__(self):
        self.lower_con = -31
        self.upper_con = 31
        self.lower_br = -111
        self.upper_br = 111

    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
        '''
        Implementation from 3D-CNN_DataGenerator repository:
        https://github.com/arehmanAzam/3D-CNN_DataGenerator/blob/master/augmentation.py
        '''
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def __call__(self, imgs):
        con = random.randint(self.lower_con, self.upper_con)
        br = random.randint(self.lower_br, self.upper_br)
        for idx in range(32):
            imgs[idx] = self.apply_brightness_contrast(imgs[idx], br, con)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """
    Normalize the values
    """

    def __init__(self, size=1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        out = (imgs / 255.)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
