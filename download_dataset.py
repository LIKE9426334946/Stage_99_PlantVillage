
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

save_dir = ROOT / "dataset" / "PlantVillage"

save_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="mohanty/PlantVillage",
    repo_type="dataset",
    local_dir=str(save_dir),
)

print("整个 PlantVillage 仓库下载完成！")
print("保存位置:", save_dir)
