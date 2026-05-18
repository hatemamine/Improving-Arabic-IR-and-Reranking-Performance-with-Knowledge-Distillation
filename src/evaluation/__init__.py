def __getattr__(name):
    if name == "compute_metrics":
        from src.evaluation.metrics import compute_metrics
        return compute_metrics
    if name == "MetricsResult":
        from src.evaluation.metrics import MetricsResult
        return MetricsResult
    if name == "PipelineEvaluator":
        from src.evaluation.evaluator import PipelineEvaluator
        return PipelineEvaluator
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
