import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor



def collate_fn(batch):
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]

    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    return input_tensor, target_tensor


def get_dataloader(data_dir, bs=1, train=True):
    path = Path(data_dir) / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    return DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
