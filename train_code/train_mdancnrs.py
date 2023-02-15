import argparse
import time
import random
import sys
import math
import json
sys.path.append("..")
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

# from datasets_utils import *
from new_div2k import Div2k
from models.mdancnrs import MDANCNRS
import numpy as np

from train_code.common import *
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler

from train_code.common import get_pretrain_model_path


# --------------------------------------------------1.训练参数设置-------------------------------------------------------##
parser = argparse.ArgumentParser()
parser.add_argument("--method_name", type=str, default="MDANCNRS", help="method/model name")
parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--hr_height", type=int, default=256, help="high res. image height 96|144|192")
# parser.add_argument("--hr_width", type=int, default=256, help="high res. image width 96|144|192")
# parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=600, help="number of batches with pixel-wise loss only")
parser.add_argument("--upscale_factor", type=int, default=2, help="")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
parser.add_argument("--seed", type=int, default=None, help="seed")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu_id")
parser.add_argument("--rgb_range", type=float, default=255., help="1 or 255")
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(opt.gpu_id)
# -----------------------随机数种子---------------------------
seed = opt.seed
if opt.seed is None:
    seed = random.randint(1, 10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ----------------------------------------------2.模型和生成效果图保存位置，训练数据集保存位置-------------------------------------------------------##
weight_path = os.path.join("../", "saved_models/", opt.method_name+'/', "x"+str(opt.upscale_factor)+'/')
os.makedirs(weight_path, exist_ok=True)
training_generate = os.path.join("../", "saved_images/", opt.method_name+"/", "x"+str(opt.upscale_factor)+"/")
os.makedirs(training_generate, exist_ok=True)
training_log_path = os.path.join("../", "saved_lines/", opt.method_name+"/", "x"+str(opt.upscale_factor)+"/")
os.makedirs(training_log_path, exist_ok=True)
training_set_path = "/media/gxx/Data2/tzc/dataset/DIV2K"

# ----------------------------------------------3.设置gpu_id和hr_shape------------------------------------------------------------------##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# hr_shape = (opt.hr_height, opt.hr_width)
hr_shape = get_higher_hr_shape(opt.upscale_factor)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# ---------------------------------------------4.定义生成模型，判别模型和损失函数，优化器------------------------------------------------##
generator = MDANCNRS(scale=opt.upscale_factor, rgb_range=opt.rgb_range).to(device)


pretrain_model_path = get_pretrain_model_path(opt.method_name, opt.upscale_factor)
if pretrain_model_path is not None:
    state_dict = torch.load(pretrain_model_path)
    generator.load_state_dict(state_dict, True)


# Losses
criterion_pixel = torch.nn.L1Loss().to(device)

# Optimizers
learning_rate = opt.lr
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(opt.b1, opt.b2))
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=300)
scheduler = lr_scheduler.StepLR(optimizer_G, step_size=200, gamma=0.5)

# ---------------------------------------------5.是否重新开始训练------------------------------------------------##
if opt.epoch != 1:
    generator.load_state_dict(torch.load(os.path.join(weight_path, "generator.pth")))

# ---------------------------------------------6.数据加载------------------------------------------------##
dataloader = DataLoader(
    Div2k(training_set_path, patch_size=hr_shape, upscale_factor=opt.upscale_factor, rgb_range=opt.rgb_range),
    batch_size=opt.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=opt.n_cpu,
)

# ---------------------------------------------7.开始训练------------------------------------------------##
"""
1.使用tensorboard绘制训练过程中的损失函数.
2.使用列表记录各个epoch的损失，用于matplotlib绘制.
3.计算训练集的psnr.
"""
# ------------------------- 1) warm_up training ---------------------------
results = {"epoch": [], "l1_loss": [], "psnr": [], "seed": seed}
for epoch in range(opt.epoch, opt.n_epochs + 1):
    scheduler.step()
    training_result = {"batch_done": 0, "l1_loss": 0.0, "mse": 0.0}
    for i, imgs in enumerate(dataloader):
        now_time = time.time()

        training_result["batch_done"] += 1

        imgs_lr = Variable(imgs["lr"].type(Tensor)).to(device)
        imgs_hr = Variable(imgs["hr"].type(Tensor)).to(device)
        gen_hr = generator(imgs_lr).to(device)

        optimizer_G.zero_grad()
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        loss_pixel.backward()
        optimizer_G.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f] [time :%f ]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item(), time.time() - now_time)
        )

        training_result["l1_loss"] += loss_pixel.item()
        batch_mse = ((gen_hr - imgs_hr) ** 2).data.mean()
        training_result["mse"] += batch_mse

    results["epoch"].append(epoch)
    results["l1_loss"].append(training_result["l1_loss"] / training_result["batch_done"])
    results["psnr"].append(-10 * math.log10(training_result["mse"] / training_result["batch_done"]))


    if epoch % opt.sample_interval == 0:
        # make grid
        # gen_hr = torch.clamp(gen_hr, 0, 1)
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.upscale_factor)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
        save_image(img_grid, os.path.join(training_generate, "%d.png" % epoch), normalize=False)

    if epoch == 300:
        torch.save(generator.state_dict(), os.path.join(weight_path, "generator300.pth"))

    if epoch % opt.checkpoint_interval == 0:
        torch.save(generator.state_dict(), os.path.join(weight_path, "generator.pth"))

        log_dir = os.path.join(training_log_path, "log.txt")
        f = open(log_dir, "w")
        f.write(json.dumps(results))
        f.close()
