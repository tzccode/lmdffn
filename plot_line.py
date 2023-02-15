import json
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--method_name", type=str, default="AWAC", help="method_name")
parser.add_argument("--upscale_factor", type=int, default=2, help="upscale_factor")
opt = parser.parse_args()


def get_log_path(method_name, upscale_factor):
    path = os.path.join("./", "saved_lines/", method_name + "/", "x" + str(upscale_factor) + "/", "log.txt")
    return path


def plot_loss_line():
    log_path = get_log_path(opt.method_name, opt.upscale_factor)
    f = open(log_path, "r")
    data = json.loads(f.read())
    epoch = data["epoch"]
    loss = data["l1_loss"]

    plt.figure()
    plt.title("loss function")
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.plot(epoch, loss, label="loss")
    plt.show()


def plot_psnr_line():
    log_path = get_log_path(opt.method_name, opt.upscale_factor)
    f = open(log_path, "r")
    data = json.loads(f.read())
    epoch = data["epoch"]
    psnr = data["psnr"]

    plt.figure()
    plt.title("psnr")
    plt.xlabel("epoch")
    plt.ylabel("psnr value")
    plt.plot(epoch, psnr, label="psnr")
    plt.show()


def plot_loss_lines():
    pass


def plot_psnr_lines():
    pass


if __name__ == "__main__":
    plot_loss_line()
    plot_psnr_line()
