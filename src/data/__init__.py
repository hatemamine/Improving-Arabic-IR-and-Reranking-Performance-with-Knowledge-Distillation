# Lazy imports to avoid hard dep failures at import time
def __getattr__(name):
    if name == "DatasetLoader":
        from src.data.loader import DatasetLoader
        return DatasetLoader
    if name == "ArabicPreprocessor":
        from src.data.preprocessor import ArabicPreprocessor
        return ArabicPreprocessor
    if name == "HardNegativeMiner":
        from src.data.hard_negatives import HardNegativeMiner
        return HardNegativeMiner
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
