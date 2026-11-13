import copy
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import torch
import yaml
from PIL import Image

import download_dataset
import model
import test as huggingface_check
import train
from checks.check_dataset import main as check_dataset
from config import dataset_path, get_device, load_config, resolve_path
from dataset import get_dataloaders


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = copy.deepcopy(load_config())
        self.config["dataset"].update({
            "root": "plants",
            "image_dir": "images/rgb",
            "train_split": "lists/fitting.txt",
            "test_split": "lists/holdout.txt",
            "image_size": [32, 32],
            "mean": [0.5, 0.25, 0.1],
            "std": [0.5, 0.5, 1.0],
        })
        self.config["model"].update(pretrained=False, num_classes=2)
        self.config["training"].update({
            "epochs": 2, "batch_size": 2, "learning_rate": 0.002,
            "weight_decay": 0.03, "device": "cpu", "output_dir": "results",
            "metrics_file": "history.csv", "checkpoint_file": "final.pth", "log_interval": 1,
        })
        self.config["checks"].update({
            "output_dir": "figures", "num_show": 1, "sample_columns": 1,
            "sample_figure_size": [2, 3], "distribution_figure_size": [3, 2],
            "dpi": 40, "show_plots": False, "check_all_images": True,
            "distribution_file": "counts.png", "samples_file": "examples.png",
        })
        self.write_config()

    def write_config(self):
        self.config_path = self.directory / "experiment.yaml"
        contents = {key: value for key, value in self.config.items() if not key.startswith("_")}
        self.config_path.write_text(yaml.safe_dump(contents), encoding="utf-8")
        self.config = load_config(self.config_path)

    def create_images(self):
        root = dataset_path(self.config, "")
        train_paths, test_paths = [], []
        for name in ("Apple___healthy", "Tomato___diseased"):
            image_dir = root / self.config["dataset"]["image_dir"] / name
            image_dir.mkdir(parents=True)
            for index in range(3):
                image_path = image_dir / f"{index}.png"
                Image.new("RGB", (20, 24), (255, 128, 0)).save(image_path)
                relative = image_path.relative_to(root).as_posix()
                (train_paths if index < 2 else test_paths).append(relative)
        for key, paths in (("train_split", train_paths), ("test_split", test_paths)):
            split_path = root / self.config["dataset"][key]
            split_path.parent.mkdir(parents=True, exist_ok=True)
            # Windows 分隔符和 UTF-8 BOM 也应能读取。
            split_path.write_text("\n".join(path.replace("/", "\\") for path in paths), encoding="utf-8-sig")

    def test_relative_paths_follow_selected_yaml(self):
        self.assertEqual(dataset_path(self.config, ""), self.directory / "plants")
        self.assertEqual(resolve_path(self.config, "results"), self.directory / "results")
        self.assertEqual(resolve_path(self.config, str(self.directory)), self.directory)

    def test_empty_and_invalid_configs_are_rejected(self):
        for section, key, value in (
            ("training", "epochs", 0),
            ("training", "num_workers", -1),
            ("training", "learning_rate", "0.001"),
            ("training", "pin_memory", "yes"),
            ("model", "pretrained", "false"),
            ("dataset", "in_channels", 1),
            ("dataset", "image_size", [32]),
            ("dataset", "std", [1, 0, 1]),
        ):
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(self.config_path)
        self.config_path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(self.config_path)

    def test_device_auto_selects_cpu_or_cuda(self):
        self.config["training"]["device"] = "auto"
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(get_device(self.config).type, "cpu")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_device(self.config).type, "cuda")

    def test_custom_dataset_paths_sizes_normalization_and_seed(self):
        self.create_images()
        first_loader, test_loader, names = get_dataloaders(self.config)
        second_loader, _, _ = get_dataloaders(self.config)
        images, labels = next(iter(first_loader))
        self.assertEqual(names, ["Apple___healthy", "Tomato___diseased"])
        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        expected = torch.tensor([1.0, (128 / 255 - 0.25) / 0.5, -0.1])
        torch.testing.assert_close(images[0, :, 0, 0], expected)
        self.assertTrue(torch.equal(labels, next(iter(second_loader))[1]))
        self.assertEqual(len(test_loader.dataset), 2)
        self.assertFalse(first_loader.pin_memory)

    def test_model_configuration_reaches_factory(self):
        for pretrained in (False, True):
            with self.subTest(pretrained=pretrained):
                self.config["model"]["pretrained"] = pretrained
                self.config["dataset"]["image_size"] = [224, 224]
                fake_model = torch.nn.Module()
                fake_model.heads = torch.nn.Sequential()
                fake_model.heads.head = torch.nn.Linear(8, 1000)
                with patch("model.vit_b_16", return_value=fake_model) as factory:
                    result = model.build_model(self.config)
                    weights = model.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
                    factory.assert_called_once_with(weights=weights, image_size=224)
                self.assertEqual(result.heads.head.out_features, 2)

    def test_incompatible_pretrained_input_fails_before_download(self):
        self.config["model"]["pretrained"] = True
        with patch("model.vit_b_16") as factory:
            with self.assertRaises(ValueError):
                model.build_model(self.config)
            factory.assert_not_called()

    def test_training_uses_yaml_and_saves_metrics_and_config(self):
        self.create_images()
        tiny_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 2))
        with patch("train.build_model", return_value=tiny_model), \
                patch("train.AdamW", wraps=train.AdamW) as optimizer, redirect_stdout(io.StringIO()):
            train.main(self.config)
        self.assertEqual(optimizer.call_args.kwargs, {"lr": 0.002, "weight_decay": 0.03})
        with (self.directory / "results/history.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
        checkpoint = torch.load(self.directory / "results/final.pth", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["config"]["training"]["learning_rate"], 0.002)
        self.assertEqual(len(checkpoint["class_names"]), 2)

    def test_remote_helpers_use_yaml_without_network(self):
        self.config["download"].update(repo_id="example/plants", name="rgb",
                                       train_split="fitting", test_split="holdout")
        with patch("download_dataset.snapshot_download") as download, redirect_stdout(io.StringIO()):
            download_dataset.main(self.config)
        download.assert_called_once_with(repo_id="example/plants", repo_type="dataset",
                                         local_dir=str(self.directory / "plants"))
        with patch("test.load_dataset", return_value={"fitting": [1], "holdout": [2]}) as load, \
                redirect_stdout(io.StringIO()):
            huggingface_check.main(self.config)
        load.assert_called_once_with(path="example/plants", name="rgb", trust_remote_code=True)

    def test_dataset_check_respects_output_and_plot_settings(self):
        self.create_images()
        with redirect_stdout(io.StringIO()):
            check_dataset(self.config)
        with Image.open(self.directory / "figures/examples.png") as image:
            self.assertEqual(image.size, (80, 120))
        self.assertTrue((self.directory / "figures/counts.png").is_file())


if __name__ == "__main__":
    unittest.main()
