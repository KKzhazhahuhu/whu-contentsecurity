import os
from torch.utils.data import Dataset
from PIL import Image


class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
            root_dir (string): 包含图像的目录路径
            transform (callable, optional): 应用于样本的可选变换
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有图像文件
        self.image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        # 遍历目录获取所有有效图像文件
        if os.path.exists(root_dir):
            for file in os.listdir(root_dir):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    self.image_files.append(os.path.join(root_dir, file))
        else:
            print(f"警告: 目录 {root_dir} 不存在")
            self.image_files = []  # 如果目录不存在，则使用空列表

        print(f"找到 {len(self.image_files)} 个图像文件。")

    def __len__(self):
        """返回数据集中样本的数量"""
        # 至少返回1，即使没有图像文件，以便进行训练
        return max(1, len(self.image_files))

    def __getitem__(self, idx):
        """
            idx (int): 样本索引
        """
        # 如果没有图像文件，则返回随机噪声（调试用）
        if len(self.image_files) == 0:
            # 创建随机噪声图像
            import numpy as np
            random_image = np.random.rand(64, 64, 3) * 255
            image = Image.fromarray(random_image.astype('uint8'))
        else:
            # 正常情况：加载图像文件
            img_path = self.image_files[idx % len(self.image_files)]
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
                # 加载失败时使用随机噪声代替
                import numpy as np
                random_image = np.random.rand(64, 64, 3) * 255
                image = Image.fromarray(random_image.astype('uint8'))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image