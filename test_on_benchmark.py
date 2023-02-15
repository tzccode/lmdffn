import argparse
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.get_model import get_model
from benchmark import BenchMark
import imageio

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--method_name", type=str, default='BPN_v2', help='please given the method name')
parser.add_argument("--test_dataset_name", type=str, default='Set5', help='please given the test dataset')
parser.add_argument("--upscale_factor", type=int, default=2, help='please given the upscale factor')
parser.add_argument("--gpu_id", type=int, default=0, help="gpu_id")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(opt.gpu_id)


def get_output_path(method_name, test_dataset_name, upscale_factor):
    """
    参数：方法名，测试集名称，放大因子。由opt给出
    返回图像输出的路径，最终由matlab程序进行测试。如果图像输出的路径不存在，则创建。
    图像输出路径例子：./test_code/SR/BI/RCAN/Set5/x2
    """
    output_dir = os.path.join('./test_code/SR/BI/', method_name+'/', test_dataset_name+'/', 'x'+str(upscale_factor))
    test_dataset_list = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
    upscale_factor_list = [2, 3, 4, 8]
    assert test_dataset_name in test_dataset_list
    assert upscale_factor in upscale_factor_list
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_model_from_method_name(method_name, upscale_factor):
    """
    1.根据method_name，也就模型的名称，来获得相应的模型。调用models.get_model中的get_model方法来获得相应的模型.
    2.根据method_name和upscale_factor找到存放模型的路径，并加载参数.
    3.将模型设置为eval()模式.
    4.返回模型
    """
    model = get_model(model_name=method_name, upscale_factor=upscale_factor)
    weight_path = os.path.join("saved_models/", method_name+'/', "x"+str(upscale_factor)+'/', "generator.pth")
    # print(weight_path)
    model.load_state_dict(torch.load(weight_path))
    print("load successfully")
    model.eval()
    return model


def get_sr_img_name(lr_image_name):
    """
    lr_image_name:bird_LRBI_x4.png
    sr_image_name:bird_RCAN_x4.png
    """
    return lr_image_name.replace("LRBI", opt.method_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------------------------------------- 1.加载模型 ----------------------------------------------- #
    # 定义模型，加载参数，设置为评估模式
    model = get_model_from_method_name(opt.method_name, opt.upscale_factor).to(device)

    # ---------------------------------------- 2.根据upscale和测试集name，读取数据 ------------------------ #
    # 读取数据在数据加载中定义
    data_loader = DataLoader(BenchMark(opt.test_dataset_name, opt.upscale_factor), batch_size=1, shuffle=False)

    # ---------------------------------------- 3.主流程，用来生成图像并保存 ------------------------------- #
    # 用模型超分，同时使用torchvision.utils.save_image保存到相应的路径供matlab测试程序来定量分析
    image_output_path = get_output_path(method_name=opt.method_name,
                                        test_dataset_name=opt.test_dataset_name,
                                        upscale_factor=opt.upscale_factor)
    test_bar = tqdm(data_loader)
    print("start generating!")
    for lr_img, lr_img_name in test_bar:
        """
        1.考虑是否已经将lr_img转为tensor, 是否将tensor转为Variable, 是否已经将lr_img标准化
        2.将lr_img输入网络中，得到超分结果
        3.考虑是否需要去标准化
        4.使用torchvision.utils.save_image保存图像。该方法需要考虑是否要去标准化，保存的是tensor.Float还是numpy.uint8
        """
        # print(lr_img_name)
        lr = Variable(lr_img)
        lr = lr.to(device)
        with torch.no_grad():
            sr_img = model(lr)
            # print(sr_img.size())
            sr_img = sr_img.clamp(0, 255).round()
        sr_img = sr_img[0].data.byte().permute(1, 2, 0).cpu()
        sr_img_name = get_sr_img_name(lr_img_name[0])
        sr_img_path = os.path.join(image_output_path, sr_img_name)
        # save_image(sr_img, sr_img_path)
        imageio.imsave(sr_img_path, sr_img)
    print("generating finish!")


if __name__ == '__main__':
    main()
