import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
import typer
from pathlib import Path
from transformers import get_scheduler

from dataset import get_dataloader
from model import build_model
from evaluate import evaluate

def compute_acc(preds, labels):
    return (preds.argmax(1) == labels).float().mean()
def train_one_epoch(model, dataloader, loss_fn, optimizer, device,writer,epoch,lr_scheduler):
    total_loss = 0
    model.train()
    pbar = tqdm(dataloader, desc="训练")
    loss_value = None
    acc_value = None
    for i,batch in enumerate(pbar):

        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        # outputs.shape: [batch_size,num_classes]
        loss = loss_fn(outputs, labels)
        acc = compute_acc(outputs, labels)
        loss_value = loss.item()*0.4 + loss_value*0.6 if loss_value else loss.item()
        acc_value = acc.item()*0.4 + acc_value*0.6 if acc_value else acc.item()
        if i%100==0:
            writer.add_scalar("train_loss", loss_value, (epoch-1)*len(dataloader)+i)
            writer.add_scalar("train_acc", acc_value, (epoch-1)*len(dataloader)+i)
        pbar.set_description(f"acc: {acc_value:.4f}, loss: {loss_value:.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()


        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(cfg_path: str):

    config = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(
        data_dir=config.data_dir, bs=config.train.batch_size, type=config.model.backbone
    )

    test_dataloader = get_dataloader(
        config.data_dir,
        bs=config.train.batch_size * 2,
        train=False,
        type=config.model.backbone,
    )

    model = build_model(config.model).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    lr_scheduler = get_scheduler(name='linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=int(len(dataloader) * 0.2),
                                 num_training_steps=int(config.train.epochs * len(dataloader)))

    writer = SummaryWriter(
        log_dir=Path(config.work_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    )

    best_acc = float(0.0)
    for epoch in range(1, config.train.epochs + 1):
        print(f"========== Epoch {epoch} ==========")
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device,writer,epoch,lr_scheduler)

        acc = evaluate(model, test_dataloader, device)

        writer.add_scalar("epoch_loss", loss, epoch)
        writer.add_scalar("val_acc", acc, epoch)
        print(f"epoch_loss: {loss:.4f}, val_acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Path(config.work_dir) / "best.pt")

        torch.save(model.state_dict(), Path(config.work_dir) / "last.pt")

    writer.close()


if __name__ == "__main__":
    typer.run(train)
