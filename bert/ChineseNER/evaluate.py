import torch
from tqdm import tqdm
import omegaconf
from pathlib import Path
import typer

from model import build_model
from dataset import get_dataloader
from predict import predict_batch


def evaluate(model, test_dataloader, device):
    total_count = 0
    correct_count = 0
    for inputs in tqdm(test_dataloader, desc='评估'):
        labels = inputs.pop('labels')
        inputs =  {k: v.to(device) for k,v in inputs.items()}

        batch_result = predict_batch(model, inputs)

        batch_result = batch_result.reshape(-1,batch_result.shape[-1])
        labels = labels.reshape(-1,)
        for result, target in zip(batch_result, labels):
            result = result.argmax().item()
            if target==-100:
                continue
            if result == target:
                correct_count += 1
            total_count += 1
    return correct_count / total_count


def run_evaluate(cfg_path:str):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 模型
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(Path(cfg.work_dir) / 'best.pt'))
    print("模型加载成功")

    # 3. 数据集
    test_dataloader = get_dataloader(cfg.data_dir,
        bs=cfg.train.batch_size,
        train=False,
        type=cfg.model.backbone,)

    # 4.评估逻辑
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc: {acc}")


if __name__ == '__main__':
    typer.run(run_evaluate)
