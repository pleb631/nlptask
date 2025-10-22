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
    # 1. 处理输入
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    # 2.预测逻辑
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_result = predict_batch(model, inputs)

    return batch_result


def run_predict(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    # 准备资源
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 分词器
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone, cache_dir=cfg.model.cache_dir
    )
    # 3. 模型
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(Path(cfg.work_dir) / "best.pt"))
    print("模型加载成功")

    print("欢迎情感分析模型(输入q或者quit退出)")

    while True:
        user_input = input("> ")
        if user_input in ["q", "quit"]:
            print("欢迎下次再来")
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue

        result = predict(user_input, model, tokenizer, device, cfg.model.seq_len)
        print(f"分类：{result.argmax()}，概率：{result.max()}")
        if result.argmax() == 1:
            print("正向评价")
        else:
            print("负向评价")


if __name__ == "__main__":
    typer.run(run_predict)
