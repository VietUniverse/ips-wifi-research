"""Discover model plugins placed in this package."""

from ips.plugins import discover


def available_models() -> list[str]:
    return sorted(discover("ips.algorithms", "MODEL_NAME", "build_model"))


def create_model(name: str, input_dim: int):
    plugins = discover("ips.algorithms", "MODEL_NAME", "build_model")
    if name not in plugins:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(sorted(plugins))}")
    return plugins[name](input_dim)
