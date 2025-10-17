from .gru import TranslationGRUModel



def modelBuildFactory(config, **kwargs):

    if config.type == "gru":
        return TranslationGRUModel(
            config=config,
            zh_vocab_size=kwargs.get("zh_vocab_size"),
            en_vocab_size=kwargs.get("en_vocab_size"),
            zh_padding_index=kwargs.get("zh_padding_index"),
            en_padding_index=kwargs.get("en_padding_index"),
        )
    else:
        raise ValueError(f"Unknown model type: {config.type}")
