import csv
import random

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import get_device, load_config, read_config_from_cli, resolve_path
from dataset import get_dataloaders
from model import build_model


def main(config=None):
    config = load_config() if config is None else config
    training = config["training"]
    random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training["seed"])
    torch.backends.cudnn.deterministic = training["deterministic"]
    torch.backends.cudnn.benchmark = not training["deterministic"]

    device = get_device(config)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    train_loader, _, class_names = get_dataloaders(config)
    if len(class_names) != config["model"]["num_classes"]:
        raise ValueError(f"数据集有 {len(class_names)} 个类别，与 model.num_classes 不一致")
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=training["learning_rate"],
                      weight_decay=training["weight_decay"])

    output_dir = resolve_path(config, training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / training["metrics_file"]
    checkpoint_path = output_dir / training["checkpoint_file"]

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy"])
        for epoch in range(training["epochs"]):
            print(f"第 {epoch + 1}/{training['epochs']} 次训练")
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                if (step + 1) % training["log_interval"] == 0:
                    print(f"Epoch [{epoch + 1}/{training['epochs']}] "
                          f"Step [{step + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            epoch_loss = total_loss / total
            epoch_accuracy = correct / total
            print(f"\nEpoch [{epoch + 1}/{training['epochs']}] "
                  f"Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}\n")
            writer.writerow([epoch + 1, epoch_loss, epoch_accuracy])
            file.flush()
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "config": {key: value for key, value in config.items() if not key.startswith("_")},
            }, checkpoint_path)

    print("训练完成！")
    print("模型保存位置:", checkpoint_path)


if __name__ == "__main__":
    main(read_config_from_cli())

