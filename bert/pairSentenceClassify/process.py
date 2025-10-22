from datasets import load_dataset, ClassLabel
from omegaconf import OmegaConf
import typer
from pathlib import Path


def process(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    print("开始处理数据")

    dataset = load_dataset(
        "json", data_files=str(Path(cfg.data_dir) / "multinli_1.0_train.jsonl")
    )["train"]

    dataset = dataset.remove_columns(
        [
            "annotator_labels",
            "genre",
            "pairID",
            "promptID",
            "sentence1_binary_parse",
            "sentence1_parse",
            "sentence2_binary_parse",
            "sentence2_parse",
        ]
    )
    dataset = dataset.rename_column("gold_label", "label")


    dataset = dataset.filter(
        lambda x: x["label"] in ["contradiction", "entailment", "neutral"]
        and x["sentence1"] is not None
        and x["sentence2"] is not None
    )
    
    dataset = dataset.cast_column(
        "label", ClassLabel(names=["contradiction", "entailment", "neutral"])
    )

    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column="label",shuffle=True)


    print(dataset_dict)

    dataset_dict.save_to_disk(cfg.data_dir)

    print("数据处理完成")


if __name__ == "__main__":
    typer.run(process)
