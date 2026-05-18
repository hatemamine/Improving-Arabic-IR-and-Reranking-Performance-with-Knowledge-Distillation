def __getattr__(name):
    if name == "BiEncoderModel":
        from src.models.bi_encoder import BiEncoderModel
        return BiEncoderModel
    if name == "CrossEncoderModel":
        from src.models.cross_encoder import CrossEncoderModel
        return CrossEncoderModel
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
