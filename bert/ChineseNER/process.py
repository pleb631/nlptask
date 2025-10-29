import typer
from datasets import Dataset
from pathlib import Path

def read_txt(txt_path):
    txt_path = Path(txt_path)
    with txt_path.open("r", encoding="utf-8") as txt_file:
        raw = txt_file.readline()
        while raw:
            yield raw
            raw = txt_file.readline()

def process_ner(example_path:str):
    data = []
    tmp_sentence = ""
    tmp_entity = []
    for raw in read_txt(example_path):
        line = raw.rstrip("\n").split()
        if len(line) > 2:
            raise ValueError(f"数据标注有误{line}")
        
        elif len(line) == 2:
            tmp_sentence += line[0]
            tmp_entity.append(line[1])
        else:
            data.append({"text": tmp_sentence, "entity": tmp_entity})
            tmp_sentence = ""
            tmp_entity = []
    
    return Dataset.from_list(data)

def process(src:str):
    src = Path(src)
    if not src.exists():
        typer.echo("数据源不存在")
        return
    train_df = process_ner(src / "china-people-daily-ner-corpus/example.train")
    test_df = process_ner(src / "china-people-daily-ner-corpus/example.test")
    dev_df = process_ner(src / "china-people-daily-ner-corpus/example.dev")

    train_df.save_to_disk(src / "train")
    test_df.save_to_disk(src / "test")
    dev_df.save_to_disk(src / "dev")


if __name__ == "__main__":
    typer.run(process)
