# Họ kNN / WKNN — hướng dẫn ghép vào repo

Họ KNN/WKNN nằm trên nhánh `feature/knn-wknn`. Nhánh này đã có mã nguồn, cấu hình, notebook Colab và test; dùng Pull Request để ghép vào `main`.

## Chạy trên Colab

1. Mở `notebooks/Colab_Train_KNN.ipynb` từ nhánh `feature/knn-wknn` và chạy ô 1. Ô này tải repo bằng Colab Secret `GITHUB_TOKEN` hoặc cho upload ZIP của nhánh nếu token thiếu/sai quyền.
2. Với fine-grained GitHub token, chọn đúng repo `VietUniverse/ips-wifi-research` và quyền **Contents: Read**; trong Colab Secrets bật quyền truy cập của notebook với `GITHUB_TOKEN`. Không dán token vào ô mã.
3. Ô 2 tự tải hai CSV UJIIndoorLoc; ô 3 dùng `knn_smoke.json` để thử, rồi đổi sang `knn_full.json` khi chạy đầy đủ. Ô 4 tải ZIP kết quả.
4. Có thể dùng `notebooks/Colab_Train_UJI.ipynb` trên cùng nhánh: chọn **Họ model → KNN/WKNN**, model và preset `knn_*.json`.

Sau khi đồng đội push commit mới, mở lại notebook từ đúng nhánh hoặc chạy lại ô tải repo. Bản notebook cũ đang mở trên Colab không tự cập nhật.

## Thuật toán và giao thức

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

Không bắt buộc: `python -m ips.experiments.knn_cli` chạy được ngay. `scipy` đã được khai báo trong `pyproject.toml`; `matplotlib` là tùy chọn để xuất biểu đồ.

## Repo giả định (đã đọc từ runner.py, chưa chạy với dữ liệu UJI thật)

- `make_building_data(train, test, building, val_size, seed, preprocess, preprocess_params)` trả về `x_train/x_val/x_test`, `y_train/y_val/y_test` (cột `LONGITUDE`, `LATITUDE`), `train_rows/val_rows/test_rows`.
- `*_rows` là vị trí dòng trong DataFrame gốc; code dùng nó để lấy cột `FLOOR`. Nếu không khớp tọa độ, chương trình dừng với lỗi **"source_row không khớp tọa độ"** thay vì chạy sai — khi đó sửa `_floor_of()` trong `knn_runner.py`.

## Khác với MLP/KAN

- Chạy bằng `knn_cli`, không qua `ips-train`, và không hiện trong `ips-train --list-options`.
- Mỗi tòa chỉ dùng train nội bộ làm tập tham chiếu (internal validation chỉ để chọn k).

## Quét số WAP (Lần 6.5)

Preset `knn_smoke.json` và `knn_full.json` có khóa `wap_sweep`. Sau phần so sánh theo tòa, runner chạy thêm
`src/ips/experiments/knn_wap_sweep.py`: xếp hạng WAP bằng Random Forest và Độ phủ chỉ trên train, giữ top-m
(m = 200 → 300), rồi đưa từng m vào kNN, WKNN 1/d, WKNN 1/d² và quét k.
Kết quả nằm trong thư mục `wap_sweep` của ZIP (`khao_sat_m_k.csv`, `tong_hop_theo_m.csv`, `chon_m.csv`,
`bang_tung_m.csv`, `m_sweep.png`) và log `[m-sweep]` trong `console_log.txt`.
Giao thức khác phần so sánh theo tòa (tham chiếu toàn bộ train, tự vote tòa, k và m chọn trên validationData.csv)
nên không so trực tiếp. Xóa khóa `wap_sweep` khỏi preset để tắt.
