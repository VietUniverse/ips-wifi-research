# WiFi indoor positioning research

Mã thí nghiệm UJIIndoorLoc cho MLP, BSRBF-KAN và BSRBF-KAN + residual KAN-Mixer. Repo này tách model, dữ liệu, bộ chạy và kết quả. Các notebook trong `notebooks/legacy/` là bản lịch sử v5–v7; số liệu cũ không tự động là kết quả của runner mới.

## Chạy trên Google Colab

1. Mở `notebooks/Colab_Train_UJI.ipynb` trên Colab và chọn GPU ở **Runtime → Change runtime type**.
2. Vì repo là private, thêm một GitHub token có quyền đọc repo vào Colab Secrets với tên `GITHUB_TOKEN`, hoặc tải ZIP của repo từ GitHub rồi upload ở ô đầu notebook. Không ghi token vào notebook, Git commit hay output.
3. Chạy các ô theo thứ tự. Notebook tải UJIIndoorLoc từ UCI vào `/content/data/uji` nếu chưa có `trainingData.csv` và `validationData.csv`.
4. Chọn một trong `configs/mlp_smoke.json`, `configs/bsrbf_baseline.json`, `configs/bsrbf_residual_mixer.json`. Smoke chạy nhanh để kiểm tra đường dữ liệu; BSRBF-KAN đầy đủ có thể mất nhiều giờ.
5. Tải ZIP kết quả từ `/content/runs`. Repo không lưu CSV gốc hoặc checkpoint.

Colab dùng cùng entry point với máy local:

```powershell
python -m pip install -e .
ips-train --config configs/mlp_smoke.json --data-dir D:\LAB\DoAnCoSo_2026\colab --output-dir runs\smoke
```

`--output-dir` phải mới hoặc rỗng. Các file chính trong mỗi run: `config.json`, `building_N/split.csv`, `building_N/history.csv`, `building_N/predictions.csv`, `building_N/checkpoint.pt`, `by_building.csv` và `summary.json`.

## Giao thức đánh giá

- `trainingData.csv` dùng để chia train/internal validation theo **nhóm vị trí**. Cùng một vị trí không ở cả hai tập.
- Chuẩn hóa tọa độ fit trên train nội bộ của từng tòa; augmentation RSS chỉ chạy trên train.
- `validationData.csv` gồm 1.111 mẫu là official test. Runner chỉ tính test sau khi chọn epoch và hệ số alpha theo internal validation.
- Từng model huấn luyện riêng theo tòa và giả định đã biết `BUILDINGID` lúc suy luận. Đây là giới hạn cần ghi trong báo cáo.
- Metric chính: mean Euclidean distance tính bằng mét trên tất cả mẫu test; xuất thêm median và P90.
- Official test đã được xem trong các thử nghiệm trước. Mọi cải tiến mới trên cùng tập này mang tính phát triển; muốn tuyên bố kết quả khách quan mới cần một bộ test chưa dùng để lựa chọn model.

Runner mới có cùng ý tưởng kiến trúc và tiền xử lý cơ bản với notebook v5, nhưng lịch học và tối ưu hóa đã được gom về cấu hình chung. **Không kỳ vọng tái hiện chính xác 8,075 m hoặc 8,154 m** nếu chưa đối chiếu từng checkpoint và điều kiện train.

## Cấu trúc

```text
src/ips/algorithms/   BSRBF-KAN, MLP, residual KAN-Mixer, model factory
src/ips/data/         UJI loader, split theo vị trí, RSS augmentation
src/ips/experiments/  train, chọn checkpoint, báo cáo, CLI
configs/              cấu hình chạy, nằm trong Git
notebooks/            Colab entry point và notebook lịch sử
tests/                kiểm tra split, AP vắng mặt, model output
runs/                 kết quả từng lần chạy, bị Git ignore
```

## Làm việc nhóm

| Nhánh | Người phụ trách | Phạm vi |
|---|---|---|
| `main` | Chủ repo | Tích hợp và release |
| `feature/data-robustness` | Thành viên 1 | AP selection, device-aware split, RSS augmentation và phân tích tòa 1/2 |
| `feature/kan-mixer` | Thành viên 2 | BSRBF-KAN, Mixer, ablation và tối ưu huấn luyện |

Hai nhánh feature được tạo từ `main`. Mỗi người clone repo, checkout nhánh của mình, commit thay đổi, push, rồi mở Pull Request vào `main`. Cần review mã và giao thức trước khi merge. Khi sửa giao diện chung trong `runner.py`, mở PR nhỏ sớm để tránh hai người sửa cùng một file lâu ngày.

```bash
git clone https://github.com/VietUniverse/ips-wifi-research.git
cd ips-wifi-research
git switch feature/data-robustness     # thành viên 1
# hoặc git switch feature/kan-mixer   # thành viên 2
git pull --ff-only
python -m unittest discover -s tests -v
git add src tests configs
git commit -m "Describe the change"
git push
```

Không commit dữ liệu, checkpoint, token hoặc notebook chứa output lớn. Mỗi PR ghi rõ **split, seed, số mẫu, metric, baseline cùng giao thức**. Check GitHub Actions phải xanh trước khi merge. Khi có username của hai thành viên, chủ repo thêm quyền collaborator; việc tạo nhánh tự nó chưa cấp quyền đọc repo private.

## Nguồn dữ liệu

[UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc). Hai file CSV được tải trực tiếp từ nguồn này trong notebook Colab. Repo chỉ chứa mã nguồn và cấu hình.
