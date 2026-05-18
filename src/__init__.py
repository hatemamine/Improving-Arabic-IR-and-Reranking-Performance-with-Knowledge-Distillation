def __getattr__(name):
    if name == "PipelineConfig":
        from src.config.config import PipelineConfig
        return PipelineConfig
    raise AttributeError(f"module 'src' has no attribute {name!r}")
