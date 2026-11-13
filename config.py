"""所有脚本共用的 YAML 配置读取和路径处理。"""

import argparse
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.yaml")


def load_config(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是非空的 YAML 映射: {config_path}")
    for section in ("model", "training", "dataset", "download", "checks"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"配置缺少有效的 {section} 分组")

    # 相对路径以配置文件所在目录为基准，不受运行时工作目录影响。
    config["_config_dir"] = str(config_path.parent)
    validate_config(config)
    return config


def validate_config(config):
    positive_integers = {
        "model": ("num_classes",),
        "training": ("epochs", "batch_size", "log_interval"),
        "checks": ("num_show", "property_samples", "leaf_records", "max_bad_examples",
                   "log_interval", "sample_columns", "dpi"),
    }
    try:
        for section, keys in positive_integers.items():
            for key in keys:
                value = config[section][key]
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{section}.{key} 必须是正整数")

        training = config["training"]
        for key in ("seed", "num_workers"):
            if type(training[key]) is not int or training[key] < 0:
                raise ValueError(f"training.{key} 必须是非负整数")
        for key, allow_zero in (("learning_rate", False), ("weight_decay", True)):
            value = training[key]
            if (type(value) not in (int, float) or not float("-inf") < value < float("inf")
                    or value < 0 or (value == 0 and not allow_zero)):
                raise ValueError(f"training.{key} 必须是有效的{'非负' if allow_zero else '正'}数值")
        if type(training["pin_memory"]) is not bool and training["pin_memory"] != "auto":
            raise ValueError("training.pin_memory 必须是 true、false 或 auto")

        for section, key in (("model", "pretrained"), ("training", "deterministic"),
                             ("download", "trust_remote_code"), ("checks", "check_all_images"),
                             ("checks", "show_plots")):
            if type(config[section][key]) is not bool:
                raise ValueError(f"{section}.{key} 必须是 true 或 false")

        dataset = config["dataset"]
        size = dataset["image_size"]
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)):
            raise ValueError("dataset.image_size 必须是两个正整数：[高度, 宽度]")
        if type(dataset["in_channels"]) is not int or dataset["in_channels"] != 3:
            raise ValueError("当前 torchvision ViT 模型要求 dataset.in_channels 为 3（RGB）")
        for key in ("mean", "std"):
            values = dataset[key]
            if (not isinstance(values, list) or len(values) != dataset["in_channels"]
                    or any(type(value) not in (int, float)
                           or not float("-inf") < value < float("inf") for value in values)):
                raise ValueError(f"dataset.{key} 必须为每个输入通道提供一个有效数值")
        if any(value <= 0 for value in dataset["std"]):
            raise ValueError("dataset.std 中的数值必须大于 0")
    except KeyError as error:
        raise ValueError(f"配置缺少必需参数: {error.args[0]}") from error


def read_config_from_cli():
    parser = argparse.ArgumentParser(description="从 YAML 文件读取项目配置")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    return load_config(parser.parse_args().config)


def resolve_path(config, value):
    """项目路径相对于所选 YAML 文件；绝对路径原样使用。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path


def dataset_path(config, value):
    """数据文件路径相对于 dataset.root。"""
    return resolve_path(config, config["dataset"]["root"]) / value


def get_device(config):
    import torch

    name = config["training"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用的 CUDA，请将 training.device 设置为 auto 或 cpu")
    return device
