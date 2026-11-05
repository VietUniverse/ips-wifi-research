"""Vẽ biểu đồ cho kết quả họ kNN mà KHÔNG sửa ips.experiments.visualize (file dùng chung).

Mục RF/HUIM là bộ chọn WAP: knn_runner chỉ ghi cột sai số của các estimator thật
(knn, wknn_*), trong khi config.json vẫn ghi models=["huim_efim"] (hoặc mục HUIM đã chọn). File này tạm thay hàm
`_models` của visualize trong lúc gọi, rồi trả lại nguyên trạng.
"""
from contextlib import contextmanager

from ips.experiments import visualize as _viz


def _models_knn(config, predictions):
    if "models" in config:
        configured = list(config["models"])
        available = [c.removesuffix("_error_m") for c in predictions
                     if c.endswith("_error_m") and c != "error_m"]
        matching = [m for m in configured if f"{m}_error_m" in predictions]
        if matching:
            return matching
        if available:
            return available
    return _original_models(config, predictions)


_original_models = _viz._models


@contextmanager
def _patched():
    _viz._models = _models_knn
    try:
        yield
    finally:
        _viz._models = _original_models


def render_run(run_dir, show=True):
    with _patched():
        return _viz.render_run(run_dir, show=show)


def compare_runs(run_dirs, labels=None, show=True):
    with _patched():
        return _viz.compare_runs(run_dirs, labels=labels, show=show)
