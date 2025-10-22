import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
import typer
from pathlib import Path

from dataset import get_dataloader
from model import build_model
from evaluate import evaluate


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for batch in tqdm(dataloader, desc="训练"):

        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        # outputs.shape: [batch_size,num_classes]
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(cfg_path: str):

    config = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(data_dir=config.data_dir, bs=config.train.batch_size)

    test_dataloader = get_dataloader(
        config.data_dir, config.train.batch_size * 2, train=False
    )

    model = build_model(config.model).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    writer = SummaryWriter(
        log_dir=Path(config.work_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    )

    best_acc = float(0.0)
    for epoch in range(1, config.train.epochs + 1):
        print(f"========== Epoch {epoch} ==========")
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        
        acc = evaluate(model, test_dataloader, device)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("Acc", acc, epoch)
        print(f"Loss: {loss:.4f}, Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Path(config.work_dir) / "best.pt")
            
        torch.save(model.state_dict(), Path(config.work_dir) / "last.pt")

    writer.close()


if __name__ == "__main__":
    typer.run(train)
