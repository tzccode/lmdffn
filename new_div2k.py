import torch.utils.data as data
import os.path
import cv2
import numpy as np
from datasets_common import *


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]


def npy_loader(path):
    return np.load(path)


IMG_EXTENSIONS = [
    '.png'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                # print(path)
                images.append(path)
    return images


class Div2k(data.Dataset):
    def __init__(self, root, patch_size, upscale_factor, rgb_range=1.):
        self.scale = upscale_factor
        self.patch_size = patch_size
        self.root = root
        self.ext = ".png"   # '.png' or '.npy'(default)
        self.n_train = 800
        self.repeat = 20
        self.rgb_range = rgb_range
        self.train = True
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data
        self.dir_hr = os.path.join(self.root, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.root, 'DIV2K_train_LR_bicubic/X' + str(self.scale))

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = set_channel(lr, hr, n_channels=3)
        lr_tensor, hr_tensor = np2Tensor(lr, hr, rgb_range=self.rgb_range)
        # return lr_tensor, hr_tensor
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self):
        if self.train:
            return self.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr
