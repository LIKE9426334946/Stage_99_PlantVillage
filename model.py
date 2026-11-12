
import torch
import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights
)


def build_model(num_classes=38, pretrained=True):

    # 加载 ViT-B/16
    weights = (
        ViT_B_16_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    model = vit_b_16(weights=weights)

    # 修改分类头
    in_features = model.heads.head.in_features

    model.heads.head = nn.Linear(
        in_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model().to(device)

    model.eval()

    # 模拟输入
    x = torch.randn(
        2, 3, 224, 224
    ).to(device)

    with torch.no_grad():

        y = model(x)

    print("Device:", device)

    print("Input shape:", x.shape)

    print("Output shape:", y.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()) / 1e6,
        "M"
    )
