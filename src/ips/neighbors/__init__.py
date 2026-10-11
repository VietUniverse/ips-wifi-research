"""Họ kNN/WKNN (không huấn luyện bằng gradient). Tách khỏi ips.algorithms để không lọt vào
factory của ips-train: model ở đây không phải torch.nn.Module.
File nào có MODEL_NAME + build_model(**params) sẽ tự được tìm thấy."""
from ips.plugins import discover


def available_models():
    return discover("ips.neighbors", "MODEL_NAME", "build_model")


def build_model(name, **params):
    models = available_models()
    if name not in models:
        raise KeyError(f"Model '{name}' không tồn tại. Có: {sorted(models)}")
    return models[name](**params)
