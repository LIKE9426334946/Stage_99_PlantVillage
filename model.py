import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from config import get_device, load_config, read_config_from_cli


def build_model(config=None):
    config = load_config() if config is None else config
    model_config = config["model"]
    if model_config["name"] != "vit_b_16":
        raise ValueError(f"当前支持的模型名称为 vit_b_16，收到: {model_config['name']}")

    height, width = config["dataset"]["image_size"]
    if height != width or height % 16:
        raise ValueError("ViT-B/16 要求输入为正方形，且边长为 16 的整数倍")
    weights = None
    if model_config["pretrained"]:
        try:
            weights = ViT_B_16_Weights[model_config["weights"]]
        except KeyError as error:
            raise ValueError(f"未知的 ViT-B/16 权重: {model_config['weights']}") from error
        required_size = weights.transforms().crop_size
        if height != required_size[0] or width != required_size[-1]:
            raise ValueError(f"所选预训练权重要求输入尺寸 {required_size}，请修改 dataset.image_size")

    model = vit_b_16(weights=weights, image_size=height)
    model.heads.head = nn.Linear(model.heads.head.in_features, model_config["num_classes"])
    return model


def main(config=None):
    config = load_config() if config is None else config
    torch.manual_seed(config["training"]["seed"])
    device = get_device(config)
    model = build_model(config).to(device)
    model.eval()
    x = torch.randn(config["training"]["batch_size"], config["dataset"]["in_channels"],
                    *config["dataset"]["image_size"], device=device)
    with torch.no_grad():
        y = model(x)
    print("Device:", device)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()) / 1e6, "M")


if __name__ == "__main__":
    main(read_config_from_cli())

