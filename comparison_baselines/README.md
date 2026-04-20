# Comparison Baselines

Thư mục này là lớp tổ chức chung cho các baseline dùng trong bài toán so sánh model.

Mục tiêu:

- gom 3 baseline chuẩn vào một chỗ dễ tra cứu
- chỉ rõ baseline nào là classical, baseline nào là deep
- map mỗi baseline tới đúng config/protocol đang dùng trong repo
- tránh việc phải nhớ thủ công nhiều đường dẫn config khác nhau

Ba baseline chuẩn hiện tại:

1. `svm_vib8`
   Baseline classical đơn giản dùng handcrafted vibration features 8-D + SVM.
2. `vibration_only_cnn`
   Baseline deep đơn cảm biến, chỉ dùng vibration.
3. `current_best_multimodal`
   Baseline mạnh hiện tại của repo, dùng vibration + temperature.

## Cấu trúc

- `definitions/`
  Mỗi file YAML mô tả một baseline chuẩn, các protocol hỗ trợ, config cần chạy, và artifact đầu ra mong đợi.

## Cách dùng

Liệt kê các baseline đã đăng ký:

```powershell
python scripts/run_comparison_baseline.py --list
```

In thông tin một baseline:

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --show
```

Chạy baseline classical theo protocol stratified:

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol stratified --action train_eval
```

Chạy baseline deep vibration-only theo protocol temporal:

```powershell
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol temporal --action train_eval
```

Chạy baseline multi-modal mạnh nhất hiện tại theo full-range và đánh giá bằng vote:

```powershell
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol fullrange --action eval --agg vote
```

## Nguyên tắc dùng cho paper

Registry này chỉ là lớp điều phối. Tính khách quan của so sánh vẫn phụ thuộc vào:

- cùng `manifest`
- cùng split logic
- cùng TTF slice
- cùng đơn vị đánh giá file-wise

Feature/model family có thể khác nhau, nhưng protocol đánh giá phải đồng nhất.
