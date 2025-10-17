import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import modelBuildFactory


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc="训练"):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]

        decoder_targets = decoder_targets.reshape(-1)

        decoder_outputs = model(encoder_inputs, decoder_inputs)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])

        loss = loss_fn(decoder_outputs, decoder_targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader()

    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / "en_vocab.txt")

    model = modelBuildFactory(
        type=config.MODEL_TYPE,
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========== Epoch {epoch} ==========")
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"Loss: {loss:.4f}")

        # 记录到Tensorboard
        writer.add_scalar("Loss", loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / "best.pt")
            print("保存模型")

    writer.close()


if __name__ == "__main__":
    train()
