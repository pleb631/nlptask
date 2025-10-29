import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import typer
from pathlib import Path

from model import build_model


def predict_batch(model, inputs):

    model.eval()
    with torch.no_grad():
        output = model(**inputs)

    batch_result = torch.softmax(output, dim=-1)
    return batch_result


def predict(text, model, tokenizer, device,entities):

    ids = tokenizer.convert_tokens_to_ids(list(text))
    input_tensor = torch.tensor(
        [[tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]],
        dtype=torch.long,
    )
    attention_mask = input_tensor != 0
    token_type_ids = torch.ones_like(input_tensor)
    inputs = {
        "input_ids": input_tensor.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
    }

    batch_result = predict_batch(model, inputs)
    batch_result = batch_result.argmax(dim=-1)
    entity = list(entities.keys())
    batch_result = [entity[i] for i in batch_result[0, 1:-1]]
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

    inputs = "丽江、大理、九寨沟、黄龙等都是涂伊想去的地方"
    
    entities = {
            "O": 0,
            "B-ORG": 1,
            "B-LOC": 2,
            "B-PER": 3,
            "I-ORG": 4,
            "I-LOC": 5,
            "I-PER": 6,
        }
    result = predict(inputs, model, tokenizer, device,entities)
    pretty_print([inputs], [result], entities)

def pretty_print(sentences, labels, entities):
    """
    labels = [['B-PER','I-PER', 'O','O','O','O','O','O','O','O','O','O','B-LOC','I-LOC','B-LOC','I-LOC','O','O','O','O'],
    ['B-LOC','I-LOC','O','B-LOC','I-LOC','O','B-LOC','I-LOC','I-LOC','O','B-LOC','I-LOC','O','O','O','B-PER','I-PER','O','O','O','O','O','O']]
    sentences=["涂伊说，如果有机会他想去赤壁看一看！",
               "丽江、大理、九寨沟、黄龙等都是涂伊想去的地方！"]
    entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}


    句子：涂伊说，如果有机会他想去黄州赤壁看一看！
    涂伊:  PER
    黄州:  LOC
    赤壁:  LOC
    句子：丽江、大理、九寨沟、黄龙等都是涂伊想去的地方！
    丽江:  LOC
    大理:  LOC
    九寨沟:  LOC
    黄龙:  LOC
    涂伊:  PER
    """

    sep_tag = [tag for tag in list(entities.keys()) if 'I' not in tag]
    result = []
    for sen, label in zip(sentences, labels):
        print(f"句子：{sen}")
        last_tag = None
        for item in zip(sen + "O", label + ['O']):
            if item[1] in sep_tag:  #
                if len(result) > 0:
                    entity = "".join(result)
                    print(f"\t{entity}:  {last_tag.split('-')[-1]}")
                    result = []
                if item[1] != 'O':
                    result.append(item[0])
                    last_tag = item[1]
            else:
                result.append(item[0])
                last_tag = item[1]


if __name__ == "__main__":
    typer.run(run_predict)
