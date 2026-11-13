from huggingface_hub import snapshot_download

from config import dataset_path, load_config, read_config_from_cli


def main(config=None):
    config = load_config() if config is None else config
    save_dir = dataset_path(config, "")
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=config["download"]["repo_id"],
                      repo_type=config["download"]["repo_type"], local_dir=str(save_dir))
    print("整个 PlantVillage 仓库下载完成！")
    print("保存位置:", save_dir)


if __name__ == "__main__":
    main(read_config_from_cli())

