from torch import nn
from transformers import AutoModel


class transformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.backbone, cache_dir=cfg.cache_dir)
        self.dropout = nn.Dropout(cfg.dropout_rate)

        self.class_num = cfg.class_num
        self.linear = nn.Linear(cfg.hidden_dim, cfg.class_num)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # shape: [batch_size, seq_len]
        output = self.bert(input_ids, attention_mask, token_type_ids)

        cls_hidden_state = output.last_hidden_state

        cls_hidden_state = self.dropout(cls_hidden_state)
        output = self.linear(cls_hidden_state)
        return output


def build_model(cfg):
    if cfg.type == "transformer":
        return transformer(cfg)
    else:
        raise ValueError("模型类型错误")
