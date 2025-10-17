import pandas as pd
from sklearn.model_selection import train_test_split

from tokenizer import EnglishTokenizer, ChineseTokenizer
import typer
from pathlib import Path

def process(src: Path, dst: Path):
    print("开始处理数据")
    # 读取文件
    df = pd.read_csv(src / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'],
                     encoding='utf-8').dropna()

    dst.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), dst / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), dst / 'en_vocab.txt')

    # 构建Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(dst / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(dst / 'en_vocab.txt')

    # 构建训练集
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 保存训练集
    train_df.to_json(dst / 'train.jsonl', orient='records', lines=True)

    # 构建测试集
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 保存测试集
    test_df.to_json(dst / 'test.jsonl', orient='records', lines=True)

    print("处理数据完成")


if __name__ == '__main__':
    typer.run(process)