from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import AutoTokenizer


class mutilpleChoiceDataset(Dataset):
    def __init__(self, path, type, cache_dir=None, train=True):
        self.data = load_from_disk(path)

        self.tokenizer = AutoTokenizer.from_pretrained(type, cache_dir=cache_dir)
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        data = self.tokenizer([
            [item["startphrase"],item["ending0"]],
            [item["startphrase"],item["ending1"]],
            [item["startphrase"],item["ending2"]],
            [item["startphrase"],item["ending3"]],],
            padding=True,

            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=512,
        )
        data["label"] = item["label"]
        return data

    def collate_fn(self, batch):
        input_ids = []
        labels = []
        token_type_ids = []
        for item in batch:
            input_ids.append(item["input_ids"].permute(1,0))
            labels.append(item["label"])
            token_type_ids.append(item["token_type_ids"].permute(1,0))

        labels = torch.tensor(labels, dtype=torch.long)

        input_tensor = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        ).permute((0,2,1))
        attention_mask = input_tensor != self.pad_token_id

        token_type_ids = pad_sequence(
            token_type_ids, batch_first=True, padding_value=self.pad_token_id
        ).permute((0,2,1))

        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


def get_dataloader(data_dir, type, cache_dir=None, train=True, bs=1):
    path = str(Path(data_dir) / ("train" if train else "validation"))
    dataset = mutilpleChoiceDataset(path, type, cache_dir)

    return DataLoader(
        dataset, batch_size=bs, shuffle=True, collate_fn=dataset.collate_fn
    )


if __name__ == "__main__":
    dataloader = get_dataloader(
        "data/",
        type="google-bert/bert-base-chinese",
        train=False,
        bs=2,
    )
    for batch in dataloader:
        for key, value in batch.items():
            print(key, "->", value.shape)
        break
