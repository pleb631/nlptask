import torch
import omegaconf
from pathlib import Path


from model import modelBuildFactory
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, en_tokenizer, max_seq_len):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入,shape:[batch_size, seq_len]
    :return: 预测结果: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    """
    model.eval()
    with torch.no_grad():

        generated = model(
            inputs,
            sos_token_index=en_tokenizer.sos_token_index,
            eos_token_index=en_tokenizer.eos_token_index,
            max_length=max_seq_len,
        )

        generated_tensor = torch.cat(generated, dim=1)

        generated_list = generated_tensor.tolist()

        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]

        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device, max_seq_len):
    # 1. 处理输入
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [batch_size, seq_len]

    # 2.预测逻辑
    batch_result = predict_batch(model, input_tensor, en_tokenizer, max_seq_len)
    return en_tokenizer.decode(batch_result[0])


def run_predict(cfg_path: str = "cfg/gru.yml"):
    # 准备资源
    # 1. 确定设备
    cfg = omegaconf.OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(Path(cfg.data_dir) / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(Path(cfg.data_dir) / "en_vocab.txt")
    print("分词器加载成功")

    model = modelBuildFactory(
        config=cfg.model,
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index,
    ).to(device)
    model.load_state_dict(torch.load(Path(cfg.work_dir) / "best.pt"))
    print("模型加载成功")

    print("欢迎使用中英翻译模型(输入q或者quit退出)")

    while True:
        user_input = input("中文：")
        if user_input in ["q", "quit"]:
            print("欢迎下次再来")
            break
        if user_input.strip() == "":
            print("请输入内容")
            continue

        result = predict(
            user_input,
            model,
            zh_tokenizer,
            en_tokenizer,
            device,
            max_seq_len=cfg.model.max_seq_len,
        )
        print("英文：", result)


if __name__ == "__main__":
    import typer

    typer.run(run_predict)
