def __getattr__(name):
    if name == "ReportGenerator":
        from src.report.reporter import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module 'src.report' has no attribute {name!r}")
