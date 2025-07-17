import argparse
import os
import torch
from models import Generator
from utils import save_images

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=1000, help="要生成的图像数量")
parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
parser.add_argument("--img_size", type=int, default=64, help="图像大小")
parser.add_argument("--channels", type=int, default=3, help="图像通道数")
parser.add_argument("--model_path", type=str, required=True, help="生成器模型路径")
parser.add_argument("--output_dir", type=str, default="generated_images", help="输出目录")
parser.add_argument("--batch_size", type=int, default=64, help="生成批次大小")
opt = parser.parse_args()

# 确保输出目录存在
os.makedirs(opt.output_dir, exist_ok=True)

# CPU设备
device = torch.device("cpu")
print(f"使用设备: {device}")

# 加载生成器模型
generator = Generator(opt.latent_dim, opt.channels, opt.img_size).to(device)
print(f"加载模型: {opt.model_path}")
generator.load_state_dict(torch.load(opt.model_path, map_location=device))
generator.eval()

# 生成图像
print(f"开始生成 {opt.n_images} 张图像...")
num_generated = 0
num_batches = (opt.n_images + opt.batch_size - 1) // opt.batch_size

with torch.no_grad():
    for i in range(num_batches):
        # 计算此批次要生成的图像数量
        current_batch_size = min(opt.batch_size, opt.n_images - num_generated)
        if current_batch_size <= 0:
            break

        # 生成随机噪声
        z = torch.randn(current_batch_size, opt.latent_dim, device=device)

        # 生成图像
        fake_imgs = generator(z)

        # 保存图像
        save_images(fake_imgs, opt.output_dir, 0, i, normalize=True)

        num_generated += current_batch_size
        print(f"已生成 {num_generated}/{opt.n_images} 张图像")

print(f"已成功生成 {num_generated} 张图像，保存在 {opt.output_dir} 目录下。")