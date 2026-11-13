from pathlib import PurePosixPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import dataset_path, get_device, load_config, read_config_from_cli


class PlantVillageDataset(Dataset):
    def __init__(self, split, transform=None, config=None):
        config = load_config() if config is None else config
        self.transform = transform
        self.image_dir = dataset_path(config, config["dataset"]["image_dir"])
        split_path = dataset_path(config, config["dataset"][f"{split}_split"])
        dataset_root = dataset_path(config, "")

        self.class_names = sorted(path.name for path in self.image_dir.iterdir() if path.is_dir())
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}

        with split_path.open("r", encoding="utf-8-sig") as file:
            image_paths = [line.strip().replace("\\", "/") for line in file if line.strip()]

        self.samples = []
        for relative_path in image_paths:
            image_path = dataset_root.joinpath(*PurePosixPath(relative_path).parts)
            class_name = image_path.relative_to(self.image_dir).parts[0]
            label = self.class_to_idx[class_name]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"数据划分文件中没有图像: {split_path}")
        print(f"{split} dataset: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_dataloaders(config=None):
    config = load_config() if config is None else config
    dataset_config = config["dataset"]
    training = config["training"]
    transform = transforms.Compose([
        transforms.Resize(tuple(dataset_config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_config["mean"], std=dataset_config["std"]),
    ])
    train_dataset = PlantVillageDataset("train", transform, config)
    test_dataset = PlantVillageDataset("test", transform, config)
    pin_memory = training["pin_memory"]
    if pin_memory == "auto":
        pin_memory = get_device(config).type == "cuda"

    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "pin_memory": pin_memory,
    }
    generator = torch.Generator().manual_seed(training["seed"])
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, test_loader, train_dataset.class_names


def main(config=None):
    train_loader, test_loader, class_names = get_dataloaders(config)
    print("\n========== Dataset Information ==========")
    print(f"Classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print("\n========== Batch Information ==========")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Labels: {labels.tolist()}")
    print("\n数据加载测试完成！")


if __name__ == "__main__":
    main(read_config_from_cli())

