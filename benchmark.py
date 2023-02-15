from torch.utils.data import Dataset
import os
from datasets_common import *
import cv2


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]


def get_test_dataset(test_dataset_name, upscale_factor):
    """
    用来判别指定的测试集名称是否正确，若正确则返回测试集LR图像的路径。
    该函数在BenchMark类中调用，用以返回LR测试集图像所在的路径。
    LR图像路径:./test_code/LR/LRBI/Set5/x2
    """
    root_dir = os.path.join('./test_code/LR/LRBI/', test_dataset_name+'/', 'x'+str(upscale_factor)+'/')
    test_dataset_list = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
    upscale_factor_list = [2, 3, 4, 8]
    assert test_dataset_name in test_dataset_list
    assert upscale_factor in upscale_factor_list
    return root_dir


class BenchMark(Dataset):
    """
    读取benchmark的LR图像，再使用训练好的模型进行超分
    LR图像路径:./test_code/LR/LRBI/Set5/x2
    """
    def __init__(self, test_dataset_name, upscale_factor, rgb_range=255.):
        self.test_dataset_path = get_test_dataset(test_dataset_name=test_dataset_name, upscale_factor=upscale_factor)
        image_list = os.listdir(self.test_dataset_path)
        for i, image_name in enumerate(image_list):
            image_list[i] = self.test_dataset_path + image_name
        self.files = image_list
        self.rgb_range = rgb_range

    def __getitem__(self, index):
        """
        将读取到的LR图像转为Tensor并标准化
        """
        lr = self._load_file(index)
        lr = set_channel(lr, n_channels=3)[0]
        lr_tensor = np2Tensor(lr, rgb_range=self.rgb_range)[0]
        img_name = self.files[index % len(self.files)].split('/')[-1]
        return lr_tensor, img_name

    def __len__(self):
        return len(self.files)

    def _load_file(self, idx):
        lr = default_loader(self.files[idx % len(self.files)])
        return lr
