# AI Guide For Classical Baselines

Mục tiêu của thư mục `classical_baselines/` là giữ một đường chạy chuẩn cho các model cổ điển sao cho:

- dùng đúng `manifest` của bài báo
- dùng đúng split logic của repo
- có thể so sánh công bằng với deep models

## File quan trọng

- `classical_baselines/train_classical.py`
  Entry point để train và evaluate baseline cổ điển.
- `classical_baselines/pipeline.py`
  Nơi ghép config, split, feature extraction, training và file-wise evaluation.
- `classical_baselines/features.py`
  Nơi định nghĩa handcrafted features cho classical models.
- `classical_baselines/configs/svm_vib8_stratified.yaml`
  Config baseline cổ điển cho split stratified.
- `classical_baselines/configs/svm_vib8_temporal.yaml`
  Config baseline cổ điển cho split temporal.

## Nguyên tắc bắt buộc

Khi sửa hoặc mở rộng classical baselines, phải giữ:

- cùng `data/manifest.csv`
- cùng labels như `datasets/logs_ttf.py`
- cùng `window_seconds` và `hop_seconds` nếu đang so sánh trực tiếp
- cùng split protocol với model deep
- cùng file-wise evaluation ở test time

Không được tạo một dataset riêng cho classical model nếu mục tiêu là đưa vào bài báo.

## Split logic

`classical_baselines/pipeline.py` đang dùng lại `LogsTTFDataset` chỉ để lấy `items` theo đúng logic chia train/val/test của repo.

Điểm quan trọng:

- không dùng transform của deep model
- không dùng temp branch của deep model
- chỉ reuse phần chọn file theo `manifest`

Như vậy split của classical và deep là cùng một split.

## Baseline hiện có

Hiện tại đã chuẩn bị baseline chạy được:

- `SVM`
- `RandomForest`
- `LogisticRegression`

Feature extractor mặc định:

- `vib_stats_8d`

`vib_stats_8d` hiện được định nghĩa là:

- `rms_x`
- `std_x`
- `peak_x`
- `crest_x`
- `rms_y`
- `std_y`
- `peak_y`
- `crest_y`

Lưu ý: đây là một baseline 8-D hợp lý và tái lập được. Nó không tự động đồng nghĩa với "đúng y hệt" mọi baseline 8D trong literature.

## Cách chạy

Stratified:

```powershell
python classical_baselines/train_classical.py --config classical_baselines/configs/svm_vib8_stratified.yaml
```

Temporal:

```powershell
python classical_baselines/train_classical.py --config classical_baselines/configs/svm_vib8_temporal.yaml
```

Artifacts sẽ được ghi vào:

- `runs/classical/.../model.pkl`
- `runs/classical/.../report_test.txt`
- `runs/classical/.../confusion_matrix_test.csv`
- `runs/classical/.../predictions_test.csv`

## Nếu AI cần thêm baseline mới

Thứ tự đúng:

1. Thêm feature extractor mới trong `classical_baselines/features.py`
2. Nếu cần, thêm model mới trong `classical_baselines/pipeline.py`
3. Tạo config mới trong `classical_baselines/configs/`
4. Giữ nguyên manifest, split và file-wise evaluation
5. Không thay protocol chỉ để tăng điểm

## Nếu AI cần khớp chặt hơn với bài báo

Nên kiểm tra lại:

- baseline `8D vib SVM` trong bản thảo đang ám chỉ đúng bộ feature nào
- có cần train ở mức window rồi aggregate theo file như hiện tại không
- hay phải train trực tiếp trên file-level handcrafted features

Nếu bài báo đã chốt một baseline classical cụ thể khác, hãy cập nhật `features.py` để khớp đúng định nghĩa đó.

## Tiêu chuẩn công bằng

So sánh classical với model mới là hợp lệ nếu:

- cùng samples
- cùng splits
- cùng metric
- cùng đơn vị đánh giá file-wise

Không cần cùng họ feature.

## Việc không nên làm

- Không lấy số từ paper khác ghép trực tiếp vào bảng kết quả của repo này.
- Không đổi split của classical nhưng giữ split cũ cho deep.
- Không đánh giá classical ở window-level rồi so với deep ở file-level.
- Không dùng `100.1%` trong text báo cáo mới; dùng `[0,100]%` và coi interval cuối là right-closed.
