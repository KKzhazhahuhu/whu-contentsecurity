import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels=3, img_size=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = channels

        # 计算初始特征图的尺寸
        self.init_size = self.img_size // 4  # 最终图像尺寸的1/4

        # 第一层：将潜在向量映射到特征图
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )

        # 卷积块：逐步上采样到目标图像尺寸
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # 将潜在向量映射到特征空间
        out = self.l1(z)
        # 重塑为卷积特征图
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # 应用卷积块生成图像
        img = self.conv_blocks(out)
        return img


class Critic(nn.Module):
    def __init__(self, channels=3, img_size=64):
        super(Critic, self).__init__()

        # 卷积块设计
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # 卷积层序列
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # 计算卷积层输出的特征图大小
        ds_size = img_size // 2 ** 4  # 经过4次下采样

        # 全连接层输出单一值
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity