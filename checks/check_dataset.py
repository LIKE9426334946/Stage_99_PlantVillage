import json
import random
from pathlib import Path, PurePosixPath
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 1. 基本配置
# ============================================================

DATASET_ROOT = Path(r"D:\04_Datasets\PlantVillage")

COLOR_DIR = DATASET_ROOT / "raw" / "color"

SPLIT_DIR = DATASET_ROOT / "splits"

LEAF_MAP_PATH = DATASET_ROOT / "leaf_grouping" / "leaf-map.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_check_outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

NUM_SHOW = 12

SEED = 42

# 是否检查所有图像是否损坏
# True: 检查全部图像，耗时较长
# False: 只检查随机展示的图像
CHECK_ALL_IMAGES = False

random.seed(SEED)


# ============================================================
# 2. 检查目录
# ============================================================


def check_directories():

    print("\n========== 目录检查 ==========")

    required_paths = [DATASET_ROOT, COLOR_DIR, SPLIT_DIR, LEAF_MAP_PATH]

    for path in required_paths:

        status = "存在" if path.exists() else "不存在"

        print(f"{status}: {path}")

    if not COLOR_DIR.exists():
        raise FileNotFoundError(f"找不到 RGB 图像目录: {COLOR_DIR}")

    if not SPLIT_DIR.exists():
        raise FileNotFoundError(f"找不到数据划分目录: {SPLIT_DIR}")


# ============================================================
# 3. 获取全部类别及图像路径
# ============================================================


def get_dataset_images():

    class_dirs = sorted([path for path in COLOR_DIR.iterdir() if path.is_dir()])

    class_to_images = {}

    for class_dir in class_dirs:

        images = sorted(
            [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        )

        class_to_images[class_dir.name] = images

    return class_to_images


# ============================================================
# 4. 读取官方训练集和测试集划分
# ============================================================


def read_split(split_name):

    split_path = SPLIT_DIR / split_name

    if not split_path.exists():
        raise FileNotFoundError(f"找不到划分文件: {split_path}")

    with open(split_path, "r", encoding="utf-8-sig") as f:

        paths = [line.strip().replace("\\", "/") for line in f if line.strip()]

    return paths


def resolve_image_path(relative_path):

    parts = PurePosixPath(relative_path).parts

    return DATASET_ROOT.joinpath(*parts)


def get_class_from_path(relative_path):

    parts = PurePosixPath(relative_path).parts

    if len(parts) >= 4 and parts[0] == "raw" and parts[1] == "color":
        return parts[2]

    return None


# ============================================================
# 5. 检查数据集基本信息
# ============================================================


def print_dataset_info(class_to_images):

    print("\n========== 数据集基本信息 ==========")

    num_classes = len(class_to_images)

    total_images = sum(len(images) for images in class_to_images.values())

    crops = {class_name.split("___")[0] for class_name in class_to_images}

    print(f"数据集名称: PlantVillage")
    print(f"图像目录: {COLOR_DIR}")
    print(f"植物种类: {len(crops)}")
    print(f"分类类别: {num_classes}")
    print(f"图像总数: {total_images}")

    print("\n植物种类:")

    for crop in sorted(crops):
        print(f"  {crop}")

    print("\n类别名称:")

    for index, class_name in enumerate(sorted(class_to_images)):
        print(f"{index:02d} | {class_name}")


# ============================================================
# 6. 统计每个类别的图像数量
# ============================================================


def print_class_distribution(class_to_images):

    print("\n========== 类别图像数量 ==========")

    counts = {}

    for class_name, images in class_to_images.items():

        counts[class_name] = len(images)

    for class_name in sorted(counts):

        print(f"{class_name:<55}" f"{counts[class_name]:>6} 张")

    values = list(counts.values())

    print("\n类别数量统计:")

    print(f"最大类别图像数: {max(values)}")
    print(f"最小类别图像数: {min(values)}")
    print(f"平均每类图像数: {sum(values) / len(values):.2f}")


# ============================================================
# 7. 检查官方数据集划分
# ============================================================


def check_splits(class_to_images):

    print("\n========== 官方数据集划分 ==========")

    train_paths = read_split("color_train.txt")

    test_paths = read_split("color_test.txt")

    print(f"训练集记录数: {len(train_paths)}")
    print(f"测试集记录数: {len(test_paths)}")

    # 检查路径是否存在

    train_valid = [path for path in train_paths if resolve_image_path(path).is_file()]

    test_valid = [path for path in test_paths if resolve_image_path(path).is_file()]

    train_missing = len(train_paths) - len(train_valid)

    test_missing = len(test_paths) - len(test_valid)

    print(f"\n训练集有效路径: {len(train_valid)}")
    print(f"测试集有效路径: {len(test_valid)}")

    print(f"训练集缺失路径: {train_missing}")
    print(f"测试集缺失路径: {test_missing}")

    # 检查重复路径

    train_set = set(train_paths)
    test_set = set(test_paths)

    overlap = train_set & test_set

    print(f"\n训练集内部重复记录: {len(train_paths) - len(train_set)}")
    print(f"测试集内部重复记录: {len(test_paths) - len(test_set)}")
    print(f"训练集与测试集交集: {len(overlap)}")

    # 检查未被划分的图像

    all_paths = {path.relative_to(DATASET_ROOT).as_posix() for images in class_to_images.values() for path in images}

    split_paths = train_set | test_set

    unlisted = all_paths - split_paths

    print(f"未出现在官方划分中的图像: {len(unlisted)}")

    # 统计每个类别的训练集和测试集数量

    train_counts = Counter(get_class_from_path(path) for path in train_valid)

    test_counts = Counter(get_class_from_path(path) for path in test_valid)

    print("\n========== 各类别划分统计 ==========")

    print(f"{'类别名称':<55}" f"{'总数':>8}" f"{'Train':>8}" f"{'Test':>8}")

    for class_name in sorted(class_to_images):

        total = len(class_to_images[class_name])

        train = train_counts[class_name]

        test = test_counts[class_name]

        print(f"{class_name:<55}" f"{total:>8}" f"{train:>8}" f"{test:>8}")

    return train_valid, test_valid


# ============================================================
# 8. 检查叶片分组信息
# ============================================================


def check_leaf_grouping():

    print("\n========== 叶片分组信息 ==========")

    if not LEAF_MAP_PATH.exists():

        print("没有找到 leaf-map.json")

        return

    with open(LEAF_MAP_PATH, "r", encoding="utf-8") as f:

        leaf_map = json.load(f)

    print(f"叶片分组映射数量: {len(leaf_map)}")

    print("随机展示 3 条记录:")

    for key in list(leaf_map)[:3]:

        print(f"{key}: {leaf_map[key]}")


# ============================================================
# 9. 检查图像尺寸和通道信息
# ============================================================


def check_image_properties(image_paths):

    print("\n========== 图像属性检查 ==========")

    if not image_paths:
        print("没有可供检查的图像")
        return

    samples = random.sample(image_paths, min(10, len(image_paths)))

    for relative_path in samples:

        image_path = resolve_image_path(relative_path)

        with Image.open(image_path) as image:

            print(
                f"{image_path.name:<65}" f"Size: {str(image.size):<15}" f"Mode: {image.mode}" f"Format: {image.format}"
            )


# ============================================================
# 10. 检查损坏图像
# ============================================================


def check_corrupted_images(image_paths):

    print("\n========== 图像完整性检查 ==========")

    bad_images = []

    for index, relative_path in enumerate(image_paths):

        image_path = resolve_image_path(relative_path)

        try:

            with Image.open(image_path) as image:
                image.verify()

        except Exception as error:

            bad_images.append((relative_path, str(error)))

        if (index + 1) % 5000 == 0:

            print(f"已检查: {index + 1}/{len(image_paths)}")

    print(f"\n检查图像总数: {len(image_paths)}")
    print(f"损坏图像数量: {len(bad_images)}")

    for path, error in bad_images[:10]:

        print(path, error)


# ============================================================
# 11. 绘制类别数量分布图
# ============================================================


def plot_class_distribution(class_to_images):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = sorted(class_to_images)

    counts = [len(class_to_images[name]) for name in class_names]

    plt.figure(figsize=(20, 8))

    plt.bar(range(len(class_names)), counts)

    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)

    plt.xlabel("Class")

    plt.ylabel("Number of Images")

    plt.title("PlantVillage Class Distribution")

    plt.tight_layout()

    save_path = OUTPUT_DIR / "class_distribution.png"

    plt.savefig(save_path, dpi=200)

    print(f"\n类别分布图已保存: {save_path}")

    plt.show()


# ============================================================
# 12. 随机展示图像
# ============================================================


def visualize_samples(train_paths):

    print("\n========== 随机图像展示 ==========")
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not train_paths:
        print("训练集没有有效图像")
        return

    samples = random.sample(train_paths, min(NUM_SHOW, len(train_paths)))

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))

    for ax in axes.flat:
        ax.axis("off")

    for ax, relative_path in zip(axes.flat, samples):

        image_path = resolve_image_path(relative_path)

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            ax.imshow(image)

        class_name = get_class_from_path(relative_path)

        ax.set_title(class_name.replace("___", "\n"), fontsize=9)

        ax.axis("off")

    plt.suptitle("PlantVillage Random Samples", fontsize=14)

    plt.tight_layout()

    save_path = OUTPUT_DIR / "random_samples.png"

    plt.savefig(save_path, dpi=200)

    print(f"随机图像已保存: {save_path}")

    plt.show()


# ============================================================
# 13. 主函数
# ============================================================


def main():

    check_directories()

    class_to_images = get_dataset_images()

    print_dataset_info(class_to_images)

    print_class_distribution(class_to_images)

    train_paths, test_paths = check_splits(class_to_images)

    check_leaf_grouping()

    check_image_properties(train_paths)

    visualize_samples(train_paths)

    plot_class_distribution(class_to_images)

    if CHECK_ALL_IMAGES:

        all_paths = [
            path
            for images in class_to_images.values()
            for path in [p.relative_to(DATASET_ROOT).as_posix() for p in images]
        ]

        check_corrupted_images(all_paths)

    print("\n========== 数据集检查结束 ==========")


if __name__ == "__main__":

    main()
