import torch
from torch import nn


class TranslationEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )

    def forward(self, x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]
        output, _ = self.gru(embed)
        # output.shape: [batch_size, seq_len, hidden_size]

        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]
        # last_hidden_state.shape: [batch_size, hidden_size]
        return last_hidden_state


class TranslationDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_0):

        embed = self.embedding(x)
        output, hidden_n = self.gru(embed, hidden_0)
        output = self.linear(output)

        return output, hidden_n


class TranslationGRUModel(nn.Module):
    def __init__(
        self, config, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index
    ):
        super().__init__()
        self.encoder = TranslationEncoder(
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            vocab_size=zh_vocab_size,
            padding_index=zh_padding_index,
        )
        self.decoder = TranslationDecoder(
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            vocab_size=en_vocab_size,
            padding_index=en_padding_index,
        )

    def forward(
        self,
        encoder_inputs,
        decoder_inputs=None,
        sos_token_index=None,
        eos_token_index=None,
        max_length=50,
    ):
        # encoder_inputs.shape: [batch_size, src_seq_len]
        # decoder_inputs.shape: [batch_size, tgt_seq_len]
        context_vector = self.encoder(encoder_inputs)
        # context_vector.shape: [batch_size, hidden_size]

        if not decoder_inputs is None:
            out, _ = self.decoder(decoder_inputs, context_vector.unsqueeze(0))
            return out
        else:

            decoder_hidden = context_vector.unsqueeze(0)
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
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                next_token_indexes = torch.argmax(decoder_output, dim=-1)
                generated.append(next_token_indexes)

                decoder_input = next_token_indexes

                is_finished |= next_token_indexes.squeeze(1) == eos_token_index
                if is_finished.all():
                    break

            return generated
