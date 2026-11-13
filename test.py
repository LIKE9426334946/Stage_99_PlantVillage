"""通过 Hugging Face 加载数据集的检查脚本。"""

from datasets import load_dataset

from config import load_config, read_config_from_cli


def main(config=None):
    config = load_config() if config is None else config
    download = config["download"]
    print("开始加载 PlantVillage...")
    dataset = load_dataset(path=download["repo_id"], name=download["name"],
                           trust_remote_code=download["trust_remote_code"])
    print("\n加载成功！")
    print(dataset)
    print("训练集数量:", len(dataset[download["train_split"]]))
    print("测试集数量:", len(dataset[download["test_split"]]))


if __name__ == "__main__":
    main(read_config_from_cli())

