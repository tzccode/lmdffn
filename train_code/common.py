import os
import torch


def get_model_output_path(method_name, upscale_factor):
    """
    用来获得模型输出的位置
    例如：../saved_models/RCAN/x4,其中RCAN为method_name
    """
    output_path = os.path.join('../', 'saved_models/', method_name+'/', 'x'+str(upscale_factor)+'/')
    return output_path


"""
def get_hr_shape(upscale_factor):
    # 根据upscale_factor获得hr图像的尺寸
    if upscale_factor == 2:
        hr_shape = (96, 96)
    elif upscale_factor == 3:
        hr_shape = (144, 144)
    elif upscale_factor == 4:
        hr_shape = (192, 192)
    elif upscale_factor == 8:
        hr_shape = (384, 384)
    else:
        raise NotImplementedError
    return hr_shape
"""

def get_hr_shape(upscale_factor):
    # 根据upscale_factor获得hr图像的尺寸
    if upscale_factor == 2:
        hr_shape = 96
    elif upscale_factor == 3:
        hr_shape = 144
    elif upscale_factor == 4:
        hr_shape = 192
    elif upscale_factor == 8:
        hr_shape = 384
    else:
        raise NotImplementedError
    return hr_shape


def get_higher_hr_shape(upscale_factor):
    # 根据upscale_factor获得hr图像的尺寸
    if upscale_factor == 2:
        hr_shape = 64 * 2
    elif upscale_factor == 3:
        hr_shape = 64 * 3
    elif upscale_factor == 4:
        hr_shape = 64 * 4
    elif upscale_factor == 8:
        hr_shape = 64 * 8
    else:
        raise NotImplementedError
    return hr_shape


"""
def get_hr_shape(upscale_factor):
    # 根据upscale_factor获得hr图像的尺寸
    if upscale_factor == 2:
        hr_shape = (128, 128)
    elif upscale_factor == 3:
        hr_shape = (192, 192)
    elif upscale_factor == 4:
        hr_shape = (256, 256)
    elif upscale_factor == 8:
        hr_shape = (384, 384)
    else:
        raise NotImplementedError
    return hr_shape
"""


def get_pretrain_model_path(method_name, upscale_factor):
    """
    x3和x4放大因子要加载x2模型的初始参数,因此返回x2模型的路径。
    参数为method_name和upscale_factor
    """
    if upscale_factor == 2:
        return None
    elif upscale_factor == 3:
        return os.path.join(get_model_output_path(method_name=method_name, upscale_factor=2), "generator.pth")
    elif upscale_factor == 4:
        return os.path.join(get_model_output_path(method_name=method_name, upscale_factor=2), "generator.pth")
    elif upscale_factor == 8:
        return os.path.join(get_model_output_path(method_name=method_name, upscale_factor=2), "generator.pth")
    else:
        raise NotImplementedError


def get_training_log_path(method_name, upscale_factor):
    """
    获取训练曲线所保存的位置
    """
    training_log_path = os.path.join("../", "saved_lines/", method_name+"/", "x"+str(upscale_factor)+"/")
    return os.path.join(training_log_path, "loss.txt")


if __name__ == '__main__':
    # print(os.listdir(get_model_output_path('RCAN', 4)))
    print(get_pretrain_model_path("BPN", 2))
    print(get_pretrain_model_path("BPN", 3))
    print(get_pretrain_model_path("BPN", 4))
