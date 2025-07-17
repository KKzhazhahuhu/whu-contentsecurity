import os
import torch
import torchvision.utils as vutils
import numpy as np


def weights_init_normal(m):
    """
    初始化模型权重
        m: 模型层
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def save_images(images, path, epoch, batch_i, normalize=True):
    """
    保存生成的图像
        images: 生成的图像批次
        path: 保存路径
        epoch: 当前训练轮数
        batch_i: 当前批次索引
        normalize: 是否归一化图像
    """
    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    # 保存整个批次的图像网格
    grid = vutils.make_grid(images, padding=2, normalize=normalize)
    save_path = os.path.join(path, f"epoch_{epoch}_batch_{batch_i}.png")
    vutils.save_image(grid, save_path, normalize=normalize)

    # 单独保存每张图像
    if "final" in path:  # 只为最终生成的图像单独保存
        img_dir = os.path.join(path, "individual")
        os.makedirs(img_dir, exist_ok=True)

        for i, img in enumerate(images):
            # 计算全局索引，确保文件名唯一
            global_idx = batch_i * images.shape[0] + i
            img_path = os.path.join(img_dir, f"image_{global_idx:04d}.png")
            vutils.save_image(img, img_path, normalize=normalize)