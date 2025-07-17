import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Generator, Critic
from dataset import AnimeDataset
from utils import save_images, weights_init_normal

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="训练的轮数")
parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
parser.add_argument("--img_size", type=int, default=64, help="图像大小")
parser.add_argument("--channels", type=int, default=3, help="图像通道数")
parser.add_argument("--n_critic", type=int, default=5, help="critic训练次数")
parser.add_argument("--sample_interval", type=int, default=400, help="保存样本的间隔")
parser.add_argument("--lambda_gp", type=float, default=10, help="梯度惩罚项的权重")
parser.add_argument("--dataset_path", type=str, default="data/anime_faces", help="数据集路径")
parser.add_argument("--output_path", type=str, default="generated_images", help="输出路径")
parser.add_argument("--model_path", type=str, default="saved_models", help="模型保存路径")
opt = parser.parse_args()

# 创建输出目录
os.makedirs(opt.output_path, exist_ok=True)
os.makedirs(opt.model_path, exist_ok=True)

device = torch.device("cpu")
print(f"使用设备: {device}")

# 初始化生成器和判别器
generator = Generator(opt.latent_dim, opt.channels, opt.img_size).to(device)
critic = Critic(opt.channels, opt.img_size).to(device)

# 初始化权重
generator.apply(weights_init_normal)
critic.apply(weights_init_normal)

# 配置数据预处理
transforms_list = [
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]
transform = transforms.Compose(transforms_list)

# 加载数据集
dataset = AnimeDataset(opt.dataset_path, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=opt.lr, betas=(0.5, 0.9))


# 计算梯度惩罚
def compute_gradient_penalty(critic, real_samples, fake_samples):
    """计算WGAN-GP中的梯度惩罚项"""
    # 随机数 alpha
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    # 在真实样本和生成样本之间进行线性插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # 计算判别器对插值样本的输出
    d_interpolates = critic(interpolates)
    # 创建一个全1的张量
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # 计算梯度的范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 训练循环
batches_done = 0
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # 真实图像
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        #  训练Critic
        optimizer_C.zero_grad()
        # 生成随机噪声
        z = torch.randn(batch_size, opt.latent_dim, device=device)
        # 生成假图像
        fake_imgs = generator(z)

        # 计算真实图像和假图像的Critic输出
        real_validity = critic(real_imgs)
        fake_validity = critic(fake_imgs.detach())

        # 计算梯度惩罚
        gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs.detach())

        # Critic损失函数
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_C.step()

        # 只有在critic完成n_critic次训练后才训练Generator
        if i % opt.n_critic == 0:
            #  训练Generator
            optimizer_G.zero_grad()

            # 生成新的假图像
            z = torch.randn(batch_size, opt.latent_dim, device=device)
            fake_imgs = generator(z)

            # 计算判别器对生成图像的输出
            fake_validity = critic(fake_imgs)

            # Generator损失函数
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

        # 保存生成的样本
        if batches_done % opt.sample_interval == 0:
            save_images(fake_imgs, opt.output_path, epoch, i)

        batches_done += 1

    # 每轮结束后保存模型
    torch.save(generator.state_dict(), os.path.join(opt.model_path, f"generator_epoch_{epoch}.pth"))
    torch.save(critic.state_dict(), os.path.join(opt.model_path, f"critic_epoch_{epoch}.pth"))

# 生成最终的1000张图像
print("生成1000张二次元漫画脸图像...")
with torch.no_grad():
    generator.eval()
    num_generated = 0
    num_batches = (1000 + opt.batch_size - 1) // opt.batch_size  # 计算需要多少批次来生成1000张图像

    for i in range(num_batches):
        # 计算这一批次要生成多少图像
        current_batch_size = min(opt.batch_size, 1000 - num_generated)
        if current_batch_size <= 0:
            break
        # 生成随机噪声
        z = torch.randn(current_batch_size, opt.latent_dim, device=device)
        # 生成图像
        fake_imgs = generator(z)
        # 保存图像
        save_images(fake_imgs, os.path.join(opt.output_path, "final"), 0, i, normalize=True)
        num_generated += current_batch_size
        print(f"已生成 {num_generated}/1000 张图像")

print(f"已成功生成1000张二次元漫画脸图像，保存在 {os.path.join(opt.output_path, 'final')} 目录下。")