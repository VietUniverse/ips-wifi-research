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

Phần quét chạy khi `models` của preset có mục `rf` hoặc một trong hai mục HUIM; tham số tùy chọn đặt trong khóa
`wap_sweep` (không bắt buộc). Sau phần so sánh theo tòa, runner chạy thêm
`src/ips/experiments/knn_wap_sweep.py`: xếp hạng WAP bằng Random Forest (mục `rf`) hoặc khai phá tập mục lợi ích
cao bằng UP-Growth (mục `huim_upgrowth`) / EFIM (mục `huim_efim`) chỉ trên train, giữ top-m (m = 10% → 90% số WAP),
rồi đưa từng m vào kNN, WKNN 1/d, WKNN 1/d² và quét k.
Kết quả nằm trong thư mục `wap_rf`, `wap_huim_upgrowth` hoặc `wap_huim_efim` của ZIP (`khao_sat_m_k.csv`,
`tong_hop_theo_m.csv`, `chon_m.csv`, `bang_tung_m.csv`, `m_sweep.png`); riêng HUIM còn có `huim_thoi_gian.csv`,
`huim_nhat_ky.csv`, `huim_top_itemset.csv`, `huim_diem_wap.csv`, `huim_thong_tin.json`. Log `[m-sweep]` nằm trong
`console_log.txt`.

### Hai mục HUIM

- Chọn một mục: chỉ thuật toán đó chạy; thư mục của mục còn lại không được tạo. Thư viện `pami` được cài tự động
  (`pip install pami`) ở lần chạy đầu.
- Chọn cả hai: runner gom vào một lần khai phá chung, dùng cùng `min_util`; nếu một bên tràn RAM/quá thời gian thì
  cả hai tăng ngưỡng và chỉ nhận khi cả hai cùng thành công. Mỗi thuật toán vẫn có thư mục kết quả đầy đủ, cộng thêm
  `wap_huim_compare/`:
  - `so_sanh_tong_hop.csv`: một dòng mỗi thuật toán (min_util, itemset, thời gian, RAM, m tốt nhất, sai số).
  - `so_sanh_thoi_gian.csv` và `so_sanh_thoi_gian.png`: thời gian khai phá, xếp hạng, quét m, xuất file và RAM đỉnh.
  - `so_sanh_min_util.csv`: từng lần leo ngưỡng của cả hai.
  - `so_sanh_chon_m.csv`: m tốt nhất và sai số theo từng phương pháp kNN/WKNN.
  - `so_sanh_thong_tin.json`: ngưỡng cuối, checksum/itemset, kết luận thuật toán nhanh hơn.
- Tham số HUIM nằm trong `wap_sweep` (`min_util_ratio`, `escalate`, `max_tries`, `n_tx`, `scale`, `unit_profit`,
  `time_limit_s`, `mem_fraction`). Khi chọn cả hai, ngưỡng và số lần thử lấy từ cấu hình chung để so sánh công bằng.
- Trong `Colab_Train_UJI`, họ **KNN/WKNN** hiển thị nhãn thân thiện cho từng mục; chọn cả hai HUIM sẽ gom vào một
  lần gọi `knn_cli` với thư mục `huim_upgrowth__huim_efim`.

Giao thức khác phần so sánh theo tòa (tham chiếu toàn bộ train, tự vote tòa, k và m chọn trên validationData.csv)
nên không so trực tiếp. Xóa khóa `wap_sweep` khỏi preset để tắt.
