"""Mục "huim_upgrowth": chọn trước tập con WAP bằng UP-Growth rồi mới đưa vào kNN/WKNN.

Đây KHÔNG phải model định vị. Chọn mục này thì runner chạy như thường (kNN, WKNN 1/d,
WKNN 1/d²) rồi chạy thêm luồng quét tỉ lệ WAP trong ips.experiments.knn_wap_sweep với
xếp hạng WAP của thuật toán này. Không chọn thì UP-Growth không chạy.

File này chỉ đăng ký thuật toán; phần khai phá dùng chung nằm trong huim.py. Muốn chạy
cả UP-Growth và EFIM trong cùng một lần, chọn thêm mục "huim_efim"; khi đó runner dùng
chung min_util và tạo thêm thư mục so sánh."""
from .huim import ALGORITHMS, HUIMSelectorBase

MODEL_NAME = "huim_upgrowth"
_META = ALGORITHMS["upgrowth"]


class HUIMUPGrowth(HUIMSelectorBase):
    algorithm = "upgrowth"
    display_name = _META["display_name"]
    color = _META["color"]
    out_dir = _META["out_dir"]


def build_model(**params):
    return HUIMUPGrowth()
