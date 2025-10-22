from .gru import TranslationGRUModel
from .transformer import TranslationTransformerModel


def modelBuildFactory(config, zh_vocab_size,en_vocab_size,zh_padding_index,en_padding_index):

    if config.type == "gru":
        return TranslationGRUModel(
            config=config,
            zh_vocab_size=zh_vocab_size,
            en_vocab_size=en_vocab_size,
            zh_padding_index=zh_padding_index,
            en_padding_index=en_padding_index,
        )
    elif config.type == "transformer":
        return TranslationTransformerModel(
            config=config,
            zh_vocab_size=zh_vocab_size,
            en_vocab_size=en_vocab_size,
            zh_padding_index=zh_padding_index,
            en_padding_index=en_padding_index,
        )
    else:
        raise ValueError(f"Unknown model type: {config.type}")
