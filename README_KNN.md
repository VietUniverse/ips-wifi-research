# Họ kNN / WKNN — hướng dẫn ghép vào repo

Gói này **chỉ thêm file mới**, không sửa file nào của repo. Chép nguyên thư mục `src/`, `configs/`, `notebooks/`, `tests/` đè vào repo (không có file trùng tên).

## Nên dán vào `README.md` (mục "Thêm model hoặc cách xử lý RSS")

> **Họ kNN/WKNN** (`src/ips/neighbors/`): kNN, WKNN (1/d), WKNN (1/d²) không phải `torch.nn.Module` và không huấn luyện bằng gradient nên nằm ngoài `ips.algorithms` và chạy bằng runner riêng, **dùng lại `load_csvs` và `make_building_data`** của repo.
>
> ```powershell
> python -m ips.experiments.knn_cli --list-options
> python -m ips.experiments.knn_cli --config configs/knn_smoke.json --data-dir D:\LAB\data --output-dir runs\knn_smoke
> python -m ips.experiments.knn_cli --config configs/knn_full.json  --data-dir D:\LAB\data --menu
> ```
>
> Giao thức: mỗi tòa một lần (biết sẵn `BUILDINGID`), **k chọn trên internal validation**, official test chỉ dùng để báo cáo; sai số = khoảng cách Euclid (m) trên **tất cả** mẫu test, kèm median, P90 và độ chính xác tầng (kNN vote tầng rồi chỉ gộp tọa độ các láng giềng cùng tầng). Kết quả: `by_building.csv`, `summary.json`, `building_N/split.csv`, `building_N/predictions.csv`, `sweep_by_k.csv`, `pairwise_same_k.csv`, `wilcoxon.csv`, `summary_table.csv` và biểu đồ `C1…C6_*.png`. Không có `checkpoint.pt` vì không có tham số học.
>
> Thêm thuật toán họ kNN: một file `.py` trong `src/ips/neighbors/` có `MODEL_NAME` và `build_model(**params)`; kế thừa `BaseLocalizer` (hoặc `InversePowerWKNN`) và chỉ viết `weights()`. Model mới tự vào menu và phần so sánh.
>
> `rss_powed` (`src/ips/data/preprocessors/rss_powed.py`) là bộ tiền xử lý theo đúng giao diện plugin (AP vắng = `-1`), dùng được cả cho model PyTorch qua `preprocess`.

## Tùy chọn: thêm lệnh `ips-knn`

Trong `pyproject.toml` của repo thêm một dòng vào `[project.scripts]`:

```toml
ips-knn = "ips.experiments.knn_cli:main"
```

Không bắt buộc: `python -m ips.experiments.knn_cli` chạy được ngay. Thư viện cần thêm: `scipy` và `matplotlib` (đi kèm `scikit-learn`/`pandas` mà runner đã dùng).

## Repo giả định (đã đọc từ runner.py, chưa chạy với dữ liệu UJI thật)

- `make_building_data(train, test, building, val_size, seed, preprocess, preprocess_params)` trả về `x_train/x_val/x_test`, `y_train/y_val/y_test` (cột `LONGITUDE`, `LATITUDE`), `train_rows/val_rows/test_rows`.
- `*_rows` là vị trí dòng trong DataFrame gốc; code dùng nó để lấy cột `FLOOR`. Nếu không khớp tọa độ, chương trình dừng với lỗi **"source_row không khớp tọa độ"** thay vì chạy sai — khi đó sửa `_floor_of()` trong `knn_runner.py`.

## Khác với MLP/KAN

- Chạy bằng `knn_cli`, không qua `ips-train`, và không hiện trong `ips-train --list-options`.
- Mỗi tòa chỉ dùng train nội bộ làm tập tham chiếu (internal validation chỉ để chọn k).
