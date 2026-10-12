# Định vị trong nhà bằng WiFi RSSI

Repo này chứa pipeline thí nghiệm UJIIndoorLoc: tải dữ liệu, tiền xử lý RSSI, huấn luyện model, đánh giá theo từng tòa và xuất biểu đồ/bảng kết quả. Colab dùng mã nền từ `main` và tự lấy plugin mới từ các nhánh `feature/*` trên GitHub; không cần tạo nhánh tích hợp hay chép code thủ công giữa các nhánh.

## Chạy trên Google Colab

### 1. Mở notebook và bật GPU

Mở [`notebooks/Colab_Train_UJI.ipynb`](notebooks/Colab_Train_UJI.ipynb) trên GitHub bằng Google Colab. Chọn **Runtime → Change runtime type → T4 GPU** nếu có. Chạy notebook từ trên xuống, bắt đầu ở ô đầu tiên.

### 2. Cấp quyền đọc repo private

Notebook cần đọc `main` và các nhánh `feature/*` qua GitHub API. Trong Colab:

1. Mở biểu tượng **Secrets** (chìa khóa) ở thanh bên.
2. Tạo secret tên `GITHUB_TOKEN`, dán token vào ô giá trị và bật quyền truy cập cho notebook.
3. Token cần quyền đọc nội dung repo private `VietUniverse/ips-wifi-research`: fine-grained token chọn đúng repo, quyền **Contents: Read-only** và **Metadata: Read-only**. Không dán token vào cell, file notebook, output hoặc Git.
4. Chạy lại ô đầu sau khi cập nhật secret.

Nếu nhận HTTP 401, token sai/hết hạn hoặc thiếu quyền đọc. HTTP 404 thường có nghĩa token chưa được cấp cho repo private hoặc sai tên repo. Notebook báo rõ lỗi và không in token.

### 3. Notebook hợp nhất mã plugin

Ô đầu tự động:

1. Tải mã nền và `configs/` từ `main`.
2. Lấy danh sách nhánh `feature/*` đang có trên GitHub.
3. Quét từng nhánh và chép các file plugin/config mới vào project Colab.
4. Giữ file từ `main` nếu cùng đường dẫn đã tồn tại, để plugin chưa merge không ghi đè runner dùng chung.
5. Cài project, sau đó menu tự phát hiện model và cách xử lý RSS.

Hiện các nhánh đóng góp gồm:

| Nhánh | Thành phần được phát hiện |
|---|---|
| `feature/knn-wknn` | kNN, WKNN nghịch đảo khoảng cách, WKNN nghịch đảo bình phương; `rss_powed`; preset và runner KNN |
| `feature/CCPos_CDAE-CNN` | plugin `cdae_cnn_ccpos` và preset của model |
| `feature/MMJ` | `sae_mmj`, `cdae_mmj`, mã dùng chung và preset |
| `main` | MLP, BSRBF-KAN, residual KAN-Mixer, RSS chuẩn hóa và AP frequency |

Tất cả nhánh `feature/*` được quét tự động. Khi thành viên push file plugin mới hoặc config mới, chỉ cần **chạy lại từ ô đầu** để Colab tải phiên bản mới. Colab đang chạy không tự cập nhật mã sau khi push.

Notebook chỉ tự thêm các file Python trong `src/ips/algorithms/`, `src/ips/data/preprocessors/`, `src/ips/neighbors/`, các runner `src/ips/experiments/knn_*.py` và JSON trong `configs/`. File trùng đường dẫn không bị ghi đè. Thay đổi vào file dùng chung như `runner.py`, `factory.py`, `pyproject.toml` hoặc một plugin đã tồn tại cần được review và merge vào `main`; nếu plugin mới cần thư viện ngoài các dependency của `main`, hãy khai báo và merge dependency đó trước.

### 4. Chọn model và tiền xử lý

Menu trong notebook đợi bạn nhập lựa chọn, không tự chạy model mặc định. Chọn họ model, model, cách xử lý RSS, preset và tham số RSS khi được hỏi.

Các bộ tiền xử lý hiện có tùy theo nhánh tải được:

- `rss_standard`: chuẩn hóa RSSI, giữ AP không phát hiện theo quy ước của repo.
- `ap_frequency`: lọc AP theo số lần xuất hiện trong tập train.
- `rss_powed`: biến đổi RSS theo lũy thừa, được thêm từ nhánh KNN/WKNN.

KNN/WKNN dùng runner riêng, còn MLP/BSRBF-KAN/SAE/CDAE dùng runner neural. Trong một lượt menu, chỉ chọn nhiều model cùng họ runner. Muốn so sánh KNN với model neural, chạy hai lượt riêng trên cùng giao thức và so sánh kết quả xuất ra. Preset `bsrbf_residual_mixer.json` yêu cầu `bsrbf_kan`, `rss_standard` và đủ 520 AP.

Ô chạy sẽ in cấu hình, số dòng dữ liệu, trạng thái GPU và tiến độ huấn luyện. Kết quả gồm thông số sai số, biểu đồ CDF, phân bố lỗi, lỗi theo tòa, lịch sử train cho model neural và ma trận nhầm lẫn tầng cho KNN/WKNN. ZIP kết quả được tải xuống từ notebook. Ma trận nhầm lẫn áp dụng cho phân loại tầng; định vị tọa độ được báo cáo bằng khoảng cách Euclid theo mét.

## Chạy local

```bash
python -m pip install -e .
ips-train --list-options
ips-train --config configs/mlp_smoke.json \
  --data-dir /path/to/UJIIndoorLoc \
  --output-dir runs/mlp_smoke
```

Trên Windows PowerShell, thay đường dẫn dữ liệu bằng thư mục chứa `trainingData.csv` và `validationData.csv`. `--output-dir` phải là thư mục mới hoặc rỗng. Các run lưu cấu hình, split, lịch sử train, dự đoán, checkpoint và `summary.json`.

## Thêm model hoặc tiền xử lý ở nhánh thành viên

### Model neural

Thêm module trong `src/ips/algorithms/` khai báo `MODEL_NAME` và `build_model(input_dim)`. Hàm builder trả về `torch.nn.Module`, đầu ra có dạng `[batch, 2]` là tọa độ đã chuẩn hóa. Tên model phải duy nhất. Thêm preset JSON vào `configs/` nếu cần thiết lập train riêng.

```python
MODEL_NAME = "my_model"

def build_model(input_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    )
```

### Tiền xử lý RSS

Thêm module trong `src/ips/data/preprocessors/` khai báo `PREPROCESS_NAME` và `build_preprocessor(**params)`. Đối tượng phải có `fit(train_frame)`, `transform(frame)` và `state_dict()`. Fit chỉ trên train nội bộ của từng tòa; `transform` trả về `np.float32` dạng `[số mẫu, số đặc trưng]`, giữ AP vắng mặt theo quy ước repo.

### Quy tắc cộng tác

1. Mỗi thành viên làm trên nhánh `feature/<ten>` đã được tạo sẵn và push commit lên đó.
2. Notebook tự quét nhánh và nạp file plugin/config mới; không tạo nhánh tổng hợp.
3. Thay đổi file chung hoặc sửa code đã có cần mở Pull Request vào `main` để review.
4. Chạy smoke/kiểm tra trong PR và ghi rõ dữ liệu, split, seed, preset, số mẫu và metric. Không commit dữ liệu gốc, token, checkpoint hoặc thư mục `runs/`.

## Giao thức đánh giá UJIIndoorLoc

- `trainingData.csv` được chia thành train và internal validation theo nhóm vị trí, để cùng một vị trí không nằm ở cả hai tập.
- Scaler tọa độ, chọn AP và chọn checkpoint chỉ được fit/tune trên train và internal validation.
- `validationData.csv` gồm 1.111 mẫu official test; chỉ dùng để báo cáo sau khi đã chọn cấu hình bằng internal validation.
- Mỗi model được huấn luyện riêng theo tòa và giả định biết `BUILDINGID` khi suy luận.
- Metric vị trí chính là mean Euclidean distance (m); báo thêm median, P90 và tỷ lệ sai số không quá 5 m.
- Official test đã được sử dụng trong các thử nghiệm trước đây. Cải thiện được chọn dựa trên cùng tập này là kết quả phát triển, không phải đánh giá độc lập mới.

Runner hiện tại không nhất thiết tái hiện chính xác các con số từ notebook lịch sử; cần ghi rõ checkpoint, seed và giao thức khi so sánh.

## Thư mục chính

```text
src/ips/algorithms/       model neural và plugin discovery
src/ips/data/             loader UJI, split, augmentation, tiền xử lý RSS
src/ips/neighbors/        kNN/WKNN (từ nhánh KNN/WKNN)
src/ips/experiments/      runner train, KNN, báo cáo và biểu đồ
configs/                  cấu hình/preset
notebooks/                notebook Colab và bản lịch sử
runs/                     kết quả cục bộ, không commit
```

## Nguồn dữ liệu

[UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc). Colab tải CSV từ UCI nếu chưa có dữ liệu trong runtime.
