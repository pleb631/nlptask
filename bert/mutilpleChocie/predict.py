import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import typer
from pathlib import Path

from model import build_model


def predict_batch(model, inputs):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入,shape:[batch_size, sql_len]
    :return: 预测结果,shape:[batch_size]
    """
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
        # output.shape: [batch_size]
    batch_result = torch.softmax(output, dim=1)
    return batch_result


def predict(text, model, tokenizer, device, seq_len):

    texts = []
    for t in text[1:]:
        texts.append([text[0], t])
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
    batch_result = predict_batch(model, inputs)

    return batch_result


def run_predict(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone, cache_dir=cfg.model.cache_dir
    )

    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(Path(cfg.work_dir) / "best.pt"))
    print("模型加载成功")

    inputs = [
        "Someone falls to the ground. Someone",
        "hikes up, but this leads to the chaos of the carriage.",
        "removes the hard mask with jutting jaw, revealing someone.",
        "remains still imprisoned in the new witches lair.",
        "wears two ropes on his ax.",
    ]
    result = predict(inputs, model, tokenizer, device, cfg.model.seq_len)
    print(f"result: {torch.argmax(result, dim=1).item()}")


if __name__ == "__main__":
    typer.run(run_predict)
