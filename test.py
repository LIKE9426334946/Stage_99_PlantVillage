from datasets import load_dataset

print("开始加载 PlantVillage...")

dataset = load_dataset(
    path="mohanty/PlantVillage",
    name="default",
    trust_remote_code=True,
)

print("\n加载成功！")
print(dataset)

print("训练集数量:", len(dataset["train"]))
print("测试集数量:", len(dataset["test"]))
