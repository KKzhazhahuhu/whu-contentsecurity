import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple


class UrbanSound8KReducer:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def reduce_dataset(
            self,
            metadata_path: str,
            original_audio_dir: str,
            reduced_audio_dir: str,
            reduction_ratio: float = 0.3,
            min_samples_per_class: int = 2
    ) -> Tuple[str, pd.DataFrame]:
        """
        从每个fold中保留一定比例的数据，并创建新的缩减数据集

        参数:
        metadata_path: 原始UrbanSound8K.csv的路径
        original_audio_dir: 原始音频数据文件夹路径
        reduced_audio_dir: 缩减后数据集的保存路径
        reduction_ratio: 要保留的数据比例，默认0.3（即保留30%）
        min_samples_per_class: 每个类别至少保留的样本数

        返回:
        tuple: (缩减后的元数据文件路径, 缩减后的元数据DataFrame)
        """
        # 创建缩减数据集的目录
        os.makedirs(reduced_audio_dir, exist_ok=True)

        # 读取原始元数据
        metadata = pd.read_csv(metadata_path)
        print(f"[INFO] 原始数据集大小: {len(metadata)} 条记录")
        print(f"[INFO] 类别分布:\n{metadata['class'].value_counts()}")

        # 创建存储缩减数据的元数据
        reduced_metadata = pd.DataFrame(columns=metadata.columns)

        # 按fold和classID进行分层抽样
        for fold in range(1, 11):  # UrbanSound8K有10个fold
            fold_dir = os.path.join(reduced_audio_dir, f'fold{fold}')
            os.makedirs(fold_dir, exist_ok=True)

            # 获取当前fold的数据
            fold_data = metadata[metadata['fold'] == fold]
            print(f"\n[INFO] Processing fold {fold}: 原始大小 {len(fold_data)} 条记录")

            # 按类别保留固定比例的数据
            for class_id in sorted(fold_data['classID'].unique()):
                class_name = fold_data[fold_data['classID'] == class_id]['class'].iloc[0]
                class_data = fold_data[fold_data['classID'] == class_id]

                # 确定要保留的样本数
                n_samples = max(
                    min_samples_per_class,
                    int(len(class_data) * reduction_ratio)
                )

                # 如果样本不足，全部保留
                if n_samples >= len(class_data):
                    keep_data = class_data
                    print(f"  [CLASS {class_name}] 样本数较少 ({len(class_data)}), 全部保留")
                else:
                    keep_data = class_data.sample(n=n_samples, random_state=self.random_seed)
                    print(f"  [CLASS {class_name}] 从 {len(class_data)} 中保留 {n_samples} 个样本")

                # 将保留的数据添加到缩减元数据中
                reduced_metadata = pd.concat([reduced_metadata, keep_data], ignore_index=True)

                # 复制保留的音频文件到新目录
                self._copy_audio_files(keep_data, original_audio_dir, fold, fold_dir)

        # 打印缩减后的统计信息
        self._print_reduction_stats(metadata, reduced_metadata)

        # 保存缩减后的元数据
        reduced_metadata_path = self._save_reduced_metadata(reduced_audio_dir, reduced_metadata)

        return reduced_metadata_path, reduced_metadata

    def _copy_audio_files(
            self,
            keep_data: pd.DataFrame,
            original_audio_dir: str,
            fold: int,
            fold_dir: str
    ) -> None:
        """复制保留的音频文件到新目录"""
        for _, row in keep_data.iterrows():
            src_file = os.path.join(original_audio_dir, f'fold{fold}', row['slice_file_name'])
            dst_file = os.path.join(fold_dir, row['slice_file_name'])

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"  [WARNING] 找不到文件 {src_file}")

    def _print_reduction_stats(
            self,
            original_metadata: pd.DataFrame,
            reduced_metadata: pd.DataFrame
    ) -> None:
        """打印缩减前后的统计信息"""
        print("\n[INFO] 缩减完成")
        print(f"[INFO] 原始数据集大小: {len(original_metadata)} 条记录")
        print(f"[INFO] 缩减后数据集大小: {len(reduced_metadata)} 条记录")
        print(f"[INFO] 缩减比例: {len(reduced_metadata) / len(original_metadata) * 100:.1f}%")

        print("\n[INFO] 原始类别分布:")
        print(original_metadata['class'].value_counts())

        print("\n[INFO] 缩减后类别分布:")
        print(reduced_metadata['class'].value_counts())

    def _save_reduced_metadata(
            self,
            reduced_audio_dir: str,
            reduced_metadata: pd.DataFrame
    ) -> str:
        """保存缩减后的元数据文件"""
        reduced_metadata_path = os.path.join(
            os.path.dirname(reduced_audio_dir),
            'UrbanSound8K_reduced.csv'
        )
        reduced_metadata.to_csv(reduced_metadata_path, index=False)
        print(f"\n[INFO] 缩减后的元数据已保存至: {reduced_metadata_path}")
        return reduced_metadata_path


if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "metadata_path": "D:/浏览器下载/UrbanSound8K/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv",
        "original_audio_dir": "D:/浏览器下载/UrbanSound8K/UrbanSound8K/UrbanSound8K/audio",
        "reduced_audio_dir": "UrbanSound8K_reduced/audio",
        "reduction_ratio": 0.3,  # 保留30%的数据
        "min_samples_per_class": 2,  # 每个类别至少保留2个样本
        "random_seed": 42
    }

    # 创建缩减器并执行缩减
    reducer = UrbanSound8KReducer(random_seed=CONFIG["random_seed"])
    reduced_metadata_path, _ = reducer.reduce_dataset(
        metadata_path=CONFIG["metadata_path"],
        original_audio_dir=CONFIG["original_audio_dir"],
        reduced_audio_dir=CONFIG["reduced_audio_dir"],
        reduction_ratio=CONFIG["reduction_ratio"],
        min_samples_per_class=CONFIG["min_samples_per_class"]
    )

    print(f"\n[SUCCESS] 缩减数据集创建完成，可以使用以下路径进行后续实验:")
    print(f"元数据文件: {reduced_metadata_path}")
    print(f"音频目录: {CONFIG['reduced_audio_dir']}")