import json
import random
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
from PIL import Image

# 同时支持 python checks/check_dataset.py 和 python -m checks.check_dataset。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import dataset_path, load_config, read_config_from_cli, resolve_path


class DatasetChecker:
    def __init__(self, config):
        self.config = config
        self.settings = config["checks"]
        self.root = dataset_path(config, "")
        self.image_dir = dataset_path(config, config["dataset"]["image_dir"])
        self.train_split = dataset_path(config, config["dataset"]["train_split"])
        self.test_split = dataset_path(config, config["dataset"]["test_split"])
        self.leaf_map_path = dataset_path(config, config["dataset"]["leaf_map"])
        self.output_dir = resolve_path(config, self.settings["output_dir"])
        self.image_extensions = {extension.lower() for extension in config["dataset"]["image_extensions"]}
        self.random = random.Random(config["training"]["seed"])

    def check_directories(self):
        print("\n========== 目录检查 ==========")
        for path in (self.root, self.image_dir, self.train_split, self.test_split, self.leaf_map_path):
            status = "存在" if path.exists() else "不存在"
            print(f"{status}: {path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"找不到图像目录: {self.image_dir}")

    def get_dataset_images(self):
        class_to_images = {}
        for class_dir in sorted(path for path in self.image_dir.iterdir() if path.is_dir()):
            class_to_images[class_dir.name] = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in self.image_extensions
            )
        if not class_to_images:
            raise ValueError(f"图像目录中没有类别子目录: {self.image_dir}")
        return class_to_images

    def read_split(self, split_path):
        with split_path.open("r", encoding="utf-8-sig") as file:
            return [line.strip().replace("\\", "/") for line in file if line.strip()]

    def resolve_image_path(self, relative_path):
        return self.root.joinpath(*PurePosixPath(relative_path).parts)

    def get_class_from_path(self, relative_path):
        image_dir = PurePosixPath(self.config["dataset"]["image_dir"])
        try:
            return PurePosixPath(relative_path).relative_to(image_dir).parts[0]
        except (ValueError, IndexError):
            return None

    def print_dataset_info(self, class_to_images):
        print("\n========== 数据集基本信息 ==========")
        crops = {class_name.split("___")[0] for class_name in class_to_images}
        print("数据集名称: PlantVillage")
        print(f"图像目录: {self.image_dir}")
        print(f"植物种类: {len(crops)}")
        print(f"分类类别: {len(class_to_images)}")
        print(f"图像总数: {sum(len(images) for images in class_to_images.values())}")
        print("\n植物种类:")
        for crop in sorted(crops):
            print(f"  {crop}")
        print("\n类别名称:")
        for index, class_name in enumerate(sorted(class_to_images)):
            print(f"{index:02d} | {class_name}")

    def print_class_distribution(self, class_to_images):
        print("\n========== 类别图像数量 ==========")
        for class_name, images in sorted(class_to_images.items()):
            print(f"{class_name:<55}{len(images):>6} 张")
        values = [len(images) for images in class_to_images.values()]
        print("\n类别数量统计:")
        print(f"最大类别图像数: {max(values)}")
        print(f"最小类别图像数: {min(values)}")
        print(f"平均每类图像数: {sum(values) / len(values):.2f}")

    def check_splits(self, class_to_images):
        print("\n========== 官方数据集划分 ==========")
        train_paths = self.read_split(self.train_split)
        test_paths = self.read_split(self.test_split)
        train_valid = [path for path in train_paths if self.resolve_image_path(path).is_file()]
        test_valid = [path for path in test_paths if self.resolve_image_path(path).is_file()]
        print(f"训练集记录数: {len(train_paths)}")
        print(f"测试集记录数: {len(test_paths)}")
        print(f"\n训练集有效路径: {len(train_valid)}")
        print(f"测试集有效路径: {len(test_valid)}")
        print(f"训练集缺失路径: {len(train_paths) - len(train_valid)}")
        print(f"测试集缺失路径: {len(test_paths) - len(test_valid)}")

        train_set, test_set = set(train_paths), set(test_paths)
        print(f"\n训练集内部重复记录: {len(train_paths) - len(train_set)}")
        print(f"测试集内部重复记录: {len(test_paths) - len(test_set)}")
        print(f"训练集与测试集交集: {len(train_set & test_set)}")
        all_paths = {
            path.relative_to(self.root).as_posix()
            for images in class_to_images.values() for path in images
        }
        print(f"未出现在官方划分中的图像: {len(all_paths - train_set - test_set)}")
        train_counts = Counter(self.get_class_from_path(path) for path in train_valid)
        test_counts = Counter(self.get_class_from_path(path) for path in test_valid)
        print("\n========== 各类别划分统计 ==========")
        print(f"{'类别名称':<55}{'总数':>8}{'Train':>8}{'Test':>8}")
        for class_name in sorted(class_to_images):
            print(f"{class_name:<55}{len(class_to_images[class_name]):>8}"
                  f"{train_counts[class_name]:>8}{test_counts[class_name]:>8}")
        return train_valid, test_valid

    def check_leaf_grouping(self):
        print("\n========== 叶片分组信息 ==========")
        if not self.leaf_map_path.exists():
            print(f"没有找到叶片分组文件: {self.leaf_map_path}")
            return
        with self.leaf_map_path.open("r", encoding="utf-8") as file:
            leaf_map = json.load(file)
        print(f"叶片分组映射数量: {len(leaf_map)}")
        print(f"展示前 {self.settings['leaf_records']} 条记录:")
        for key in list(leaf_map)[:self.settings["leaf_records"]]:
            print(f"{key}: {leaf_map[key]}")

    def check_image_properties(self, image_paths):
        print("\n========== 图像属性检查 ==========")
        if not image_paths:
            print("没有可供检查的图像")
            return
        samples = self.random.sample(image_paths, min(self.settings["property_samples"], len(image_paths)))
        for relative_path in samples:
            image_path = self.resolve_image_path(relative_path)
            with Image.open(image_path) as image:
                print(f"{image_path.name:<65}Size: {str(image.size):<15}"
                      f"Mode: {image.mode} Format: {image.format}")

    def check_corrupted_images(self, image_paths):
        print("\n========== 图像完整性检查 ==========")
        bad_images = []
        for index, relative_path in enumerate(image_paths):
            try:
                with Image.open(self.resolve_image_path(relative_path)) as image:
                    image.verify()
            except Exception as error:
                bad_images.append((relative_path, str(error)))
            if (index + 1) % self.settings["log_interval"] == 0:
                print(f"已检查: {index + 1}/{len(image_paths)}")
        print(f"\n检查图像总数: {len(image_paths)}")
        print(f"损坏图像数量: {len(bad_images)}")
        for path, error in bad_images[:self.settings["max_bad_examples"]]:
            print(path, error)

    def save_figure(self, figure, filename):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.output_dir / filename
        figure.savefig(save_path, dpi=self.settings["dpi"])
        print(f"图像已保存: {save_path}")
        if self.settings["show_plots"]:
            plt.show()
        plt.close(figure)

    def plot_class_distribution(self, class_to_images):
        class_names = sorted(class_to_images)
        counts = [len(class_to_images[name]) for name in class_names]
        figure, axis = plt.subplots(figsize=self.settings["distribution_figure_size"])
        axis.bar(range(len(class_names)), counts)
        axis.set_xticks(range(len(class_names)), class_names,
                        rotation=self.settings["tick_rotation"], fontsize=self.settings["tick_font_size"])
        axis.set_xlabel("Class")
        axis.set_ylabel("Number of Images")
        axis.set_title("PlantVillage Class Distribution")
        figure.tight_layout()
        self.save_figure(figure, self.settings["distribution_file"])

    def visualize_samples(self, train_paths):
        print("\n========== 随机图像展示 ==========")
        if not train_paths:
            print("训练集没有有效图像")
            return
        samples = self.random.sample(train_paths, min(self.settings["num_show"], len(train_paths)))
        columns = self.settings["sample_columns"]
        rows = (len(samples) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=self.settings["sample_figure_size"], squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, relative_path in zip(axes.flat, samples):
            with Image.open(self.resolve_image_path(relative_path)) as image:
                axis.imshow(image.convert("RGB"))
            class_name = self.get_class_from_path(relative_path)
            axis.set_title((class_name or relative_path).replace("___", "\n"),
                           fontsize=self.settings["sample_font_size"])
        figure.suptitle("PlantVillage Random Samples", fontsize=self.settings["title_font_size"])
        figure.tight_layout()
        self.save_figure(figure, self.settings["samples_file"])


def main(config=None):
    config = load_config() if config is None else config
    checker = DatasetChecker(config)
    checker.check_directories()
    class_to_images = checker.get_dataset_images()
    checker.print_dataset_info(class_to_images)
    checker.print_class_distribution(class_to_images)
    train_paths, _ = checker.check_splits(class_to_images)
    checker.check_leaf_grouping()
    checker.check_image_properties(train_paths)
    checker.visualize_samples(train_paths)
    checker.plot_class_distribution(class_to_images)
    if config["checks"]["check_all_images"]:
        all_paths = [
            path.relative_to(checker.root).as_posix()
            for images in class_to_images.values() for path in images
        ]
        checker.check_corrupted_images(all_paths)
    print("\n========== 数据集检查结束 ==========")


if __name__ == "__main__":
    main(read_config_from_cli())

