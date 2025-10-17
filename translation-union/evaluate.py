import torch
from nltk.translate.bleu_score import corpus_bleu
import omegaconf
from pathlib import Path


from model import modelBuildFactory
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer


def evaluate(model, test_dataloader, device, en_tokenizer, max_seq_len):
    predictions = []
    # predictions: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    references = []
    # references: [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*]]]
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        # targets: [[sos,*,*,*,*,*,eos],[sos,*,*,*,*,eos,pad],[sos,*,*,*,eos,pad,pad]]
        batch_result = predict_batch(model, inputs, en_tokenizer, max_seq_len)
        # batch_result: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]

        predictions.extend(batch_result)
        references.extend(
            [
                [target[1 : target.index(en_tokenizer.eos_token_index)]]
                for target in targets
            ]
        )
    return corpus_bleu(references, predictions)


def run_evaluate(cfg_path: str = "cfg/gru.yml"):
    # 准备资源
    # 1. 确定设备
    cfg = omegaconf.OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.词表
    zh_tokenizer = ChineseTokenizer.from_vocab(Path(cfg.data_dir) / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(Path(cfg.data_dir) / "en_vocab.txt")
    print("词表加载成功")

    # 3. 模型
    model = modelBuildFactory(
        config=cfg.model,
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index,
    ).to(device)
    model.load_state_dict(torch.load(Path(cfg.work_dir) / "best.pt"))
    print("模型加载成功")

    # 4. 数据集
    test_dataloader = get_dataloader(
        cfg.data_dir, bs=cfg.training.batch_size, train=False
    )
    # 5.评估逻辑
    bleu = evaluate(model, test_dataloader, device, en_tokenizer, cfg.model.max_seq_len)
    print("评估结果")
    print(f"bleu: {bleu}")


if __name__ == "__main__":
    import typer

    typer.run(run_evaluate)
