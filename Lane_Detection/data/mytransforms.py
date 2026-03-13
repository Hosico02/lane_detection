import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import pdb
import cv2


# ===============================img transforms============================

class Compose2(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


# 图像和掩膜大小处理
class FreeScale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        mask = mask.resize((self.size[1], self.size[0]), Image.NEAREST)
        return img, mask


# 掩膜大小处理
class FreeScaleMask(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        mask = mask.resize((self.size[1], self.size[0]), Image.NEAREST)
        return mask


# 大小处理
class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


# 随机旋转
class RandomRotate(object):
    """
    作物给定的PIL。随机位置的图像具有给定大小的区域
    size可以是元组（target_height，target_width）或整数，在这种情况下，目标将是正方形（size，size）
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        assert label is None or image.size == label.size
        # 生成[-x,x]的随机数
        angle = random.randint(0, self.angle * 2) - self.angle
        #
        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        return image, label


# ===============================label transforms============================

# 规范化
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# 转换为张量
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def find_start_pos(row_sample, start_line):
    l, r = 0, len(row_sample) - 1
    while True:
        mid = int((l + r) / 2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid


# 随机偏移量
class RandomLROffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size     # 获取长宽

        img = np.array(img)     # 将图片展开
        if offset > 0:  # 向右平移
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:  # 向左平移
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)     # 将标签展开
        if offset > 0:  # 向右平移
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:  # 向左平移
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        return Image.fromarray(img), Image.fromarray(label)


# 向上下平移
class RandomUDoffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size     # 获取长宽

        img = np.array(img)     # 将图片展开
        if offset > 0:  # 向上平移
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:  # 向下平移
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)     # 将标签展开
        if offset > 0:  # 向上平移
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:  # 向下平移
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        return Image.fromarray(img), Image.fromarray(label)
