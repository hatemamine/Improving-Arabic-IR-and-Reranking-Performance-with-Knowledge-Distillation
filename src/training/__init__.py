def __getattr__(name):
    if name == "BiEncoderTrainer":
        from src.training.bi_encoder_trainer import BiEncoderTrainer
        return BiEncoderTrainer
    if name == "CrossEncoderTrainer":
        from src.training.cross_encoder_trainer import CrossEncoderTrainer
        return CrossEncoderTrainer
    raise AttributeError(f"module 'src.training' has no attribute {name!r}")
