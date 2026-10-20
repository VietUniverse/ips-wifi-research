# Định vị trong nhà bằng WiFi RSSI

Repo này chứa pipeline WiFi fingerprinting cho UJIIndoorLoc và các bộ benchmark khác: chọn dataset, chuẩn hóa RSSI, huấn luyện model và xuất biểu đồ/bảng kết quả. Colab dùng mã nền từ `main` và tự lấy plugin mới từ các nhánh `feature/*` trên GitHub; không cần tạo nhánh tích hợp hay chép code thủ công giữa các nhánh.

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

### Chọn dataset

Ở cell dữ liệu đầu notebook, chọn `UJIIndoorLoc (UCI)` để tải UJI, hoặc chọn một bộ trong ZIP research pack. Với lựa chọn từ ZIP, upload file `drive-download-...zip` khi Colab mở hộp thoại. ZIP được đọc và chuyển đổi trong runtime; repo chỉ chứa adapter, không chứa bản dữ liệu thô.

| Dataset trong ZIP | Train | Test | AP gốc |
|---|---:|---:|---:|
| HCXY (SODIndoorLoc) | 11,370 | 860 | 347 |
| SYL (SODIndoorLoc) | 8,880 | 1,020 | 363 |
| TUJI1 | 6,752 | 2,147 | 310 |
| HUS (test case 1 hoặc 2) | 20,500 | 205 | 5 |
| DLULIB | 13,000 | 6,500 | 17 |

Số dòng lấy từ ZIP bạn cung cấp. Dataset có ít hơn 520 AP được đệm bằng giá trị missing để tương thích với KAN-Mixer; với HUS và DLULIB, đầu vào thực chỉ có 5 hoặc 17 AP nên cần diễn giải riêng. UJI_LIB_DB chưa được đưa vào menu: ZIP có 620 AP và chưa có mô tả đủ rõ về đơn vị tọa độ để so sánh sai số theo mét. CETC331 không có trong ZIP này. Không gộp sai số trung bình giữa các dataset khác nhau.

### Các model KAN theo bài KSE 2025

Notebook cung cấp năm họ kiến trúc được bài báo so sánh: KAN gốc, FastKAN, BSRBF-KAN, S-KAN và FC-KAN. KAN-Mixer là mở rộng thử nghiệm của repo: giữ backbone KAN đã chọn và cho phép chọn riêng loại mạng trộn trên mỗi trục. Trong menu, chọn một model có Mixer rồi chọn một hoặc nhiều cấu hình: KAN token + KAN channel, KAN token + MLP channel, MLP token + KAN channel, hoặc MLP token + MLP channel. Với lựa chọn KAN, notebook dùng đúng họ backbone đã chọn. Mỗi cấu hình ghi riêng config, checkpoint, metrics và biểu đồ; số tham số của Mixer được in trong log để so sánh.

Bài báo gốc: [KSE 2025 proceedings](https://kse2025.kse-conferences.org/wp-content/uploads/sites/10/2025/10/kse-2025-proceedings.pdf), bài “A Comparative Study of Kolmogorov-Arnold Network Models for Robust WiFi Fingerprinting-based Positioning”. Các layer trong repo được cài dựa trên mô tả kiến trúc trong bài báo: B-spline cho KAN, Gaussian RBF cho FastKAN, kết hợp hai basis cho BSRBF-KAN, cổng chọn theo node cho S-KAN, và tương tác tích cho FC-KAN ở tầng ẩn nhỏ để kiểm soát chi phí. Bài báo không cung cấp mã nguồn thực nghiệm đi kèm, vì vậy S-KAN/FC-KAN và các tổ hợp Mixer ở đây là các hiện thực nghiên cứu theo mô tả, không được khẳng định là tái tạo từng chi tiết mã gốc. Quy trình tối ưu của runner cũng là pipeline hiện có của repo, không phải sao chép nguyên xi protocol của bài báo.

Các kết quả KAN-Mixer trước đây trong repo dùng BSRBF-KAN; menu tiếp tục giữ lựa chọn này dưới tên `KAN-Mixer (BSRBF-KAN + Residual Mixer)`. Có thêm `KAN-Mixer (KAN gốc / B-spline)`, FastKAN, S-KAN và FC-KAN + Mixer. Cần so sánh bằng cùng cách chia train/validation/test và báo cáo dữ liệu/preset/seed khi viết kết luận.

### KAN-Mixer sau chọn AP bằng Random Forest

Preset `configs/bsrbf_rf128_mixer_ap_robust.json` thêm bước chọn 128 AP bằng `RandomForestRegressor.feature_importances_`, sau khi RSS đã được chuẩn hóa. Forest và thứ hạng AP chỉ fit trên tập train của từng tòa; validation dùng chọn checkpoint/alpha, official test chỉ dùng báo cáo cuối. Runner lưu toàn bộ thứ hạng vào `rf_ap_importance.csv`, danh sách AP đã chọn vào checkpoint và vẽ `figures/rf_ap_importance.png`. Mixer nhận số AP còn lại qua phép chiếu RSS sang 64 chiều, nên có thể chạy với đầu vào đã giảm từ 520 xuống 128.

Trong Colab, chọn `KAN-Mixer (BSRBF-KAN + Residual Mixer)` và `rss_ap_robust`; bộ chọn preset tự dùng preset RF trên. Kích thước 128 là một điểm thử nghiệm, không phải giá trị tối ưu đã được xác nhận. So sánh nên chạy thêm top-k khác và baseline 520 AP với cùng seed/split trước khi kết luận.

Preset `configs/bsrbf_mixer_ap_tokens.json` là biến thể kiến trúc thử nghiệm: mỗi AP giữ một token riêng, đầu vào token gồm RSS đã chuẩn hóa và cờ AP có được quan sát hay không; embedding AP giữ danh tính cột, đặc trưng KAN được đưa vào như ngữ cảnh, sau đó Mixer xử lý cả trục AP (token mixing) và trục channel. Chọn preset này cho KAN-Mixer với `rss_ap_robust`. Kiến trúc cũ `legacy_16x8` vẫn là mặc định để các run/checkpoint trước tiếp tục tương thích. Trong Colab, mỗi cấu hình token/channel được huấn luyện độc lập; validation chọn epoch và hệ số residual, test chỉ dùng báo cáo cuối. `mixer_isolate_rng` giữ trạng thái ngẫu nhiên sau huấn luyện mixer để lần chạy nhiều tòa nhà không làm đổi khởi tạo backbone giữa các thí nghiệm. Hiện chưa có kết quả UJI đầy đủ cho bốn tổ hợp; cần chạy và đánh giá trước khi kết luận tổ hợp nào tốt hơn.

### 3. Notebook hợp nhất mã plugin

Ô đầu hỏi nhánh nguồn (Enter để dùng `main`). Khi thử thay đổi đang nằm trong PR, nhập tên nhánh PR, ví dụ `VietUniverse-patch-2`; sau khi PR được merge, dùng `main`. Ô này tự động:

1. Tải mã nền và `configs/` từ nhánh đã chọn.
2. Lấy danh sách nhánh `feature/*` đang có trên GitHub.
3. Quét từng nhánh `feature/*` và chép các file plugin/config mới vào project Colab.
4. Giữ file từ nhánh nguồn đã chọn nếu cùng đường dẫn đã tồn tại, để plugin chưa merge không ghi đè runner dùng chung.
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
- `rss_global_relative`: trộn mức RSS chuẩn hóa toàn cục với RSS tương đối so với median của từng fingerprint.
- `rss_row_rank`: thay RSS phát hiện được bằng thứ hạng AP trong từng fingerprint, ít nhạy với dịch chuyển gain đơn điệu của thiết bị.
- `rss_ap_robust`: scale từng AP theo quantile fit trên train; AP ít quan sát/ít biến thiên có thể được mask thành missing nhưng vẫn giữ nguyên 520 cột cho Mixer.

Ba cách xử lý cuối cùng là các thử nghiệm đối chứng, không mặc định tốt hơn `rss_standard`. Trong menu Colab, chọn **KAN-Mixer (BSRBF-KAN + Residual Mixer)**; notebook sẽ chỉ hiện các preset Mixer như `bsrbf_mixer_global_relative.json`, `bsrbf_mixer_row_rank.json` và `bsrbf_mixer_ap_robust.json`. Các preset bật sampling cân bằng vị trí và tăng device augmentation; kiểm tra validation trước khi xem official test. Thống kê AP chỉ fit trên train nội bộ. Cần merge thay đổi runner và notebook vào `main` trước khi Colab tải được các trường cấu hình mới.

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
- UJI có lệch thiết bị giữa train và official test; so sánh PHONEID có thể dùng để chẩn đoán domain shift, nhưng không dùng nhãn/metric official test để chọn chuẩn hóa hay checkpoint.

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
