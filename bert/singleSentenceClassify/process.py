from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import typer
from pathlib import Path


def process(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    print("开始处理数据")

    dataset = load_dataset(
        "csv", data_files=str(Path(cfg.data_dir) / "online_shopping_10_cats.csv")
    )["train"]

    dataset = dataset.remove_columns(["cat"])
    dataset = dataset.filter(lambda x: x["review"] is not None)

    dataset = dataset.cast_column("label", ClassLabel(names=["negative", "positive"]))
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column="label",shuffle=True)
    print(dataset_dict)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone, cache_dir=cfg.model.cache_dir
    )

    def batch_encode(batch):
        inputs = tokenizer(
            batch["review"], return_token_type_ids=False, return_attention_mask=False,
            truncation=True
        )
        inputs["labels"] = batch["label"]
        return inputs

    dataset_dict = dataset_dict.map(
        batch_encode, batched=True, remove_columns=["review", "label"]
    )

    dataset_dict.save_to_disk(cfg.data_dir)

    print("数据处理完成")


if __name__ == "__main__":
    typer.run(process)
