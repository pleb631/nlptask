from datasets import load_from_disk
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch],dtype=torch.long)
    input_tensor = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_tensor != 0)
    token_type_ids = torch.ones_like(input_tensor)
    return {
        'input_ids': input_tensor,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    }
def get_dataloader(data_dir,bs=3,train=True):
    path = str(Path(data_dir) / ('train' if train else 'test'))
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')
    return DataLoader(dataset, batch_size=bs, shuffle=True,collate_fn=collate_fn)

if __name__ == '__main__':
    dataloader = get_dataloader('data/',train=False)
    for batch in dataloader:
        print(batch)
