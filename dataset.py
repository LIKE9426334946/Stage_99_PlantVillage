from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# 1. 基本配置
# ============================================================

DATASET_ROOT = Path(r"D:\04_Datasets\PlantVillage")

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0


# ============================================================
# 2. Dataset
# ============================================================


class PlantVillageDataset(Dataset):

    def __init__(self, split, transform=None):

        self.transform = transform

        # 图像目录
        self.image_dir = DATASET_ROOT / "raw" / "color"

        # 划分文件
        split_path = DATASET_ROOT / "splits" / f"color_{split}.txt"

        # 获取全部类别
        self.class_names = sorted([path.name for path in self.image_dir.iterdir() if path.is_dir()])

        # 类别名称映射为整数
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}

        # 读取图像路径
        with open(split_path, "r", encoding="utf-8-sig") as f:

            image_paths = [line.strip() for line in f if line.strip()]

        self.samples = []

        for relative_path in image_paths:

            parts = PurePosixPath(relative_path).parts

            # 获取类别名称
            class_name = parts[2]

            # 获取类别编号
            label = self.class_to_idx[class_name]

            # 获取图像绝对路径
            image_path = DATASET_ROOT.joinpath(*parts)

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            self.samples.append((image_path, label))

        print(f"{split} dataset: {len(self.samples)} images")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        # 读取 RGB 图像
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        # 图像预处理
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 3. DataLoader
# ============================================================


def get_dataloaders():

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PlantVillageDataset(split="train", transform=transform)

    test_dataset = PlantVillageDataset(split="test", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_dataset.class_names


# ============================================================
# 4. 测试数据加载
# ============================================================

if __name__ == "__main__":

    train_loader, test_loader, class_names = get_dataloaders()

    print("\n========== Dataset Information ==========")

    print(f"Classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 获取一个 Batch
    images, labels = next(iter(train_loader))

    print("\n========== Batch Information ==========")

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")

    print(f"First 10 labels: {labels[:10].tolist()}")

    print("\n数据加载测试完成！")
