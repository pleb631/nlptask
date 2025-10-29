from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import AutoTokenizer


class chineseNERDataset(Dataset):
    def __init__(self, path, type, cache_dir=None, train=True):
        self.data = load_from_disk(path)

        self.tokenizer = AutoTokenizer.from_pretrained(type, cache_dir=cache_dir)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_position_embeddings = 512
        self.entities = {
            "O": 0,
            "B-ORG": 1,
            "B-LOC": 2,
            "B-PER": 3,
            "I-ORG": 4,
            "I-LOC": 5,
            "I-PER": 6,
        }
        self.ignore_idx = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text, entity = item["text"], item["entity"]
        if len(text) > self.max_position_embeddings - 2:
            text = text[: self.max_position_embeddings - 2]
            entity = entity[: self.max_position_embeddings - 2]
        entity = (
            [self.ignore_idx] + [self.entities[i] for i in entity] + [self.ignore_idx]
        )
        data = {}

        ids = self.tokenizer.convert_tokens_to_ids(list(text))
        data["input_ids"] = torch.tensor(
            [[self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]],
            dtype=torch.long,
        )

        data["label"] = torch.tensor(entity, dtype=torch.long)

        return data

    def collate_fn(self, batch):
        input_ids = [item["input_ids"].squeeze() for item in batch]
        labels = [item["label"] for item in batch]
        input_tensor = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_idx)
        attention_mask = input_tensor != 0
        token_type_ids = torch.ones_like(input_tensor)
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


def get_dataloader(data_dir, type, cache_dir=None, train=True, bs=1):
    path = str(Path(data_dir) / ("train" if train else "dev"))
    dataset = chineseNERDataset(path, type, cache_dir)

    return DataLoader(
        dataset, batch_size=bs, shuffle=True, collate_fn=dataset.collate_fn
    )


if __name__ == "__main__":
    dataloader = get_dataloader(
        "data/",
        type="google-bert/bert-base-chinese",
        train=True,
        bs=1,
    )
    for batch in dataloader:
        if batch["input_ids"].shape[1] !=batch["labels"].shape[1]:
            for key, value in batch.items():
                print(key, "->", value.shape)
            break
        # break
        ...
