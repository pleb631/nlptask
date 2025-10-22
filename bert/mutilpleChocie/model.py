from torch import nn
from transformers import AutoModel


class transformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cfg.backbone, cache_dir=cfg.cache_dir)
        self.dropout = nn.Dropout(cfg.dropout_rate)

        self.linear = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # shape: [batch_size, choice_num, hidden_dim]
        b,n,c= input_ids.shape
        input_ids = input_ids.reshape(b*n,c)
        attention_mask = attention_mask.reshape(b*n,c)
        token_type_ids = token_type_ids.reshape(b*n,c)

        output = self.bert(input_ids, attention_mask, token_type_ids)

        last_hidden_state = output.last_hidden_state

        cls_hidden_state = last_hidden_state[:, 0, :]
        cls_hidden_state = self.dropout(cls_hidden_state)
        output = self.linear(cls_hidden_state)

        output = output.reshape(b,n)

        return output


def build_model(cfg):
    if cfg.type == "transformer":
        return transformer(cfg)
    else:
        raise ValueError("模型类型错误")
