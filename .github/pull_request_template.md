## Thay đổi

<!-- Model, dữ liệu, hay quy trình đánh giá? -->

## Kiểm tra

<!-- Lệnh test và kết quả. Nếu có số sai số, nêu split, seed, số mẫu và metric. -->

## Kiểm tra rò rỉ dữ liệu

- [ ] Chỉ fit scaler/chọn AP/tune hyperparameter bằng train và internal validation.
- [ ] Official test không dùng để chọn checkpoint hay cấu hình.
- [ ] Không commit CSV dữ liệu, checkpoint, token hay kết quả `runs/`.
