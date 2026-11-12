import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import get_dataloaders
from model import build_model

# =========================
# 1. 配置
# =========================

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 训练
# =========================


def main():

    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 加载数据
    train_loader, _, class_names = get_dataloaders()

    # 加载模型
    model = build_model(num_classes=len(class_names), pretrained=True).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 保存训练记录
    metrics_path = OUTPUT_DIR / "metrics.csv"

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["epoch", "train_loss", "train_accuracy"])

        # =========================
        # 3. Epoch 循环
        # =========================

        for epoch in range(EPOCHS):
            print(f"第{EPOCHS+1}次训练")
            model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for step, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                labels = labels.to(device)

                # 清空梯度
                optimizer.zero_grad(set_to_none=True)

                # 前向传播
                outputs = model(images)

                # 计算 Loss
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 统计指标
                total_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

                if (step + 1) % 100 == 0:

                    print(
                        f"Epoch [{epoch + 1}/{EPOCHS}] "
                        f"Step [{step + 1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

            # 计算 Epoch 指标
            epoch_loss = total_loss / total

            epoch_accuracy = correct / total

            print(f"\nEpoch [{epoch + 1}/{EPOCHS}] " f"Loss: {epoch_loss:.4f} " f"Accuracy: {epoch_accuracy:.4f}\n")

            # 写入 CSV
            writer.writerow([epoch + 1, epoch_loss, epoch_accuracy])

            f.flush()

            # 保存当前模型
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "class_names": class_names},
                OUTPUT_DIR / "last.pth",
            )

    print("训练完成！")
    print("模型保存位置:", OUTPUT_DIR / "last.pth")


if __name__ == "__main__":
    main()
