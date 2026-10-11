"""Khung chung của họ kNN: vote tầng -> trọng số -> gộp tọa độ. Lớp con chỉ viết `weights()`."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from .formulas import Formulas


@dataclass
class Prediction:
    floor: np.ndarray
    x: np.ndarray
    y: np.ndarray
    n_d0: int = 0          # số mẫu có láng giềng trùng khít (d = 0)


class BaseLocalizer(ABC):
    """Không có bước huấn luyện: fit() chỉ ghi nhớ CSDL tham chiếu (train nội bộ của một tòa)."""
    display_name = ""
    color = "#999999"

    def fit(self, floor, x, y, n_floors):
        self.rf, self.rx, self.ry, self.nf = floor, x, y, n_floors
        return self

    @abstractmethod
    def weights(self, dists, mask):
        """Trả (w, n_d0); mỗi hàng w có tổng = 1."""

    def predict(self, indices, dists, k) -> Prediction:
        nb, d = indices[:, :k], dists[:, :k]
        _, pf, mask = Formulas.vote_building_floor(np.zeros_like(self.rf[nb]), self.rf[nb], 1, self.nf)
        w, n0 = self.weights(d, mask)
        px, py = Formulas.weighted_position(w, self.rx[nb], self.ry[nb])
        return Prediction(pf, px, py, n0)


class InversePowerWKNN(BaseLocalizer):
    power = 1

    def weights(self, dists, mask):
        return Formulas.inverse_power_weights(dists, mask, self.power)
