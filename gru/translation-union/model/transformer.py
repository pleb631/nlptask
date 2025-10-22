import math

import torch
from torch import nn


# class PositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#
#         pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # pos.shape: (max_len, 1)
#         _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)  # _2i.shape: (d_model/2,)
#         div_term = torch.pow(10000, _2i / self.d_model)
#
#         sins = torch.sin(pos / div_term)  # sins.shape: (max_len, d_model/2)
#         coss = torch.cos(pos / div_term)  # coss.shape: (max_len, d_model/2)
#
#         pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)  # pe.shape: (max_len, d_model)
#
#         pe[:, 0::2] = sins
#         pe[:, 1::2] = coss
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len]


class PositionEncoding(nn.Module):
    def __init__(self, max_len, dim_model):
        super().__init__()
        pe = torch.zeros([max_len, dim_model], dtype=torch.float)
        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / dim_model)))
                pe[pos, _2i + 1] = math.cos(pos / (10000 ** (_2i / dim_model)))

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, dim_model]
        seq_len = x.shape[1]
        part_pe = self.pe[0:seq_len]
        # part_pe.shape: [seq_len, dim_model]
        return x + part_pe


class TranslationTransformerModel(nn.Module):
    def __init__(
        self, config, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index
    ):
        super().__init__()
        embedding_dim = config.embedding_dim
        num_encoder_layers = config.num_encoder_layers
        num_decoder_layers = config.num_decoder_layers
        max_seq_len = config.max_seq_len
        num_heads = config.num_heads

        self.zh_embedding = nn.Embedding(
            num_embeddings=zh_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=zh_padding_index,
        )

        self.en_embedding = nn.Embedding(
            num_embeddings=en_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=en_padding_index,
        )

        # 位置编码
        self.position_encoding = PositionEncoding(max_seq_len, embedding_dim)

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=embedding_dim, out_features=en_vocab_size)

    def encode(self, src, src_pad_mask):
        # src.shape = [batch_size, src_len]
        # src_pad_mask.shape = [batch_size, src_len]
        embed = self.zh_embedding(src)
        # embed.shape = [batch_size, src_len, dim_model]
        embed = self.position_encoding(embed)

        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        # memory.shape: [batch_size, src_len, d_model]

        return memory

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        # tgt.shape: [batch_size, tgt_len]
        embed = self.en_embedding(tgt)
        embed = self.position_encoding(embed)
        # embed.shape: [batch_size, tgt_len, dim_model]

        output = self.transformer.decoder(
            tgt=embed,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        # output.shape: [batch_size, tgt_len, dim_model]

        outputs = self.linear(output)
        # outputs.shape: [batch_size, tgt_len, en_vocab_size]
        return outputs

    def forward(
        self,
        encoder_inputs,
        decoder_inputs=None,
        sos_token_index=None,
        eos_token_index=None,
        max_length=50,
    ):

        src_pad_mask = encoder_inputs == self.zh_embedding.padding_idx

        memory = self.encode(encoder_inputs, src_pad_mask)

        if not decoder_inputs is None:
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                decoder_inputs.shape[1]
            )
            decoder_outputs = self.decode(
                decoder_inputs, memory, tgt_mask, src_pad_mask
            )
            return decoder_outputs
        else:

            decoder_input = torch.full(
                [encoder_inputs.shape[0], 1],
                sos_token_index,
                device=encoder_inputs.device,
            )

            batch_size = encoder_inputs.shape[0]
            device = encoder_inputs.device
            is_finished = torch.full([batch_size], False, device=device)
            generated = []

            for i in range(max_length):
                tgt_mask = self.transformer.generate_square_subsequent_mask(
                    decoder_input.shape[1]
                )
                decoder_output = self.decode(
                    decoder_input, memory, tgt_mask, src_pad_mask
                )
                next_token_indexes = torch.argmax(
                    decoder_output[:, -1, :], dim=-1, keepdim=True
                )
                generated.append(next_token_indexes)

                decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)

                is_finished |= next_token_indexes.squeeze(1) == eos_token_index
                if is_finished.all():
                    break

            return generated
