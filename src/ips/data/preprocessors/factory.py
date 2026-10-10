"""Discover preprocessing plugins placed in this package."""

from ips.plugins import discover


def available_preprocessors() -> list[str]:
    return sorted(discover("ips.data.preprocessors", "PREPROCESS_NAME", "build_preprocessor"))


def create_preprocessor(name: str, **params):
    plugins = discover("ips.data.preprocessors", "PREPROCESS_NAME", "build_preprocessor")
    if name not in plugins:
        raise ValueError(f"Unknown preprocessor {name!r}; available: {', '.join(sorted(plugins))}")
    return plugins[name](**params)
