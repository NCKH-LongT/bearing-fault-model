# Tutorial: từ code đến cập nhật LaTeX cho bản update mới

Tài liệu này hướng dẫn quy trình thực tế cho repo này: từ lúc có một model/config mới, chạy train/eval, đối sánh với baseline, lấy số liệu đúng, cập nhật hình, và sửa `main.tex` để ra một bản paper update nhất quán.

Tài liệu này được viết dựa trên:

- `main.tex`
- `BASELINE_COMPARISON_GUIDE.md`
- `classical_baselines/AI_GUIDE.md`
- `train_logs.py`
- `eval_logs.py`
- `scripts/aggregate_ablation.py`
- `scripts/auto_train_eval.py`
- các config trong `configs/` và `configs/ablation/`

## 1. Mục tiêu của quy trình update

Mỗi lần có model mới, cần trả lời được 3 câu hỏi:

1. Model mới được train/eval theo đúng protocol của repo chưa?
2. Model mới có vượt các baseline cần thiết không?
3. Kết quả mới cần cập nhật vào những phần nào của `main.tex`?

Nếu chưa trả lời rõ 3 câu hỏi này thì không nên sửa paper.

## 2. Hiểu nhanh pipeline hiện tại của repo

Paper hiện tại đang mô tả pipeline:

- Đầu vào vibration 2 trục -> STFT -> spectrogram 2 kênh
- Nhiệt độ -> feature compact 6D
- Late fusion giữa vibration embedding và temp embedding
- Train 2 pha:
  - stratified pre-training
  - temporal fine-tuning
- Đánh giá file-wise, ưu tiên `mean-logit`, một số kết quả full-range có thêm `majority vote`

Trong code:

- Train: `train_logs.py`
- Eval: `eval_logs.py`
- Split logic: `datasets/logs_ttf.py`
- Classical baselines: `classical_baselines/`
- Tổng hợp ablation: `scripts/aggregate_ablation.py`
- Tự động thử nhiều vòng train: `scripts/auto_train_eval.py`

## 3. Các config chính đang map với paper

Đây là 3 config bạn nên hiểu trước khi update:

- Stratified dev:
  - `configs/logs_stft_train_strat.yaml`
  - out dir: `runs/logs_stft_strat`
- Temporal deployment:
  - `configs/logs_stft_full_temporal.yaml`
  - out dir: `runs/logs_stft_temporal`
- Full-range temporal:
  - `configs/logs_stft_full_temporal_alltest.yaml`
  - out dir: `runs/logs_stft_temporal_alltest`

Baseline vibration-only trong ablation:

- `configs/ablation/logs_stft_train_strat_vib.yaml`
- `configs/ablation/logs_stft_temporal_vib_ft.yaml`
- `configs/ablation/logs_stft_temporal_vib_ft_85100.yaml`

Baseline cổ điển hiện đã có pipeline riêng:

- `classical_baselines/train_classical.py`
- `classical_baselines/pipeline.py`
- `classical_baselines/features.py`
- `classical_baselines/configs/svm_vib8_stratified.yaml`
- `classical_baselines/configs/svm_vib8_temporal.yaml`

Config multi-modal gắn với bản paper hiện tại:

- `configs/logs_stft_train_strat.yaml`
- `configs/logs_stft_full_temporal.yaml`
- `configs/logs_stft_full_temporal_alltest.yaml`
- một số variant ablation trong `configs/ablation/` như `level2_tempctxdiff`

## 4. Bước 1: tạo bản update trong code

Nếu bạn có model mới, hãy xác định nó là update ở mức nào:

- Đổi backbone
- Đổi cách xử lý STFT
- Đổi temp feature
- Đổi fusion head
- Đổi split/aggregation
- Đổi hyperparameter

Nguyên tắc:

- Nếu đổi kiến trúc hoặc feature, tạo config mới thay vì sửa đè lên config cũ
- Đặt `log.out_dir` riêng để không đè kết quả mới lên run cũ
- Nếu cần fine-tune từ checkpoint cũ, cập nhật `train.init_from`

Ví dụ:

- Stratified mới: `configs/logs_stft_train_strat_myupdate.yaml`
- Temporal mới: `configs/logs_stft_full_temporal_myupdate.yaml`
- Alltest mới: `configs/logs_stft_full_temporal_alltest_myupdate.yaml`

## 5. Bước 2: chạy stratified trước

Mục đích của bước này:

- kiểm tra model train ổn định không
- lấy kết quả dev 3 lớp đầy đủ
- có checkpoint để fine-tune temporal

Lệnh chạy cơ bản:

```powershell
python train_logs.py --config configs/logs_stft_train_strat_myupdate.yaml
python eval_logs.py --config configs/logs_stft_train_strat_myupdate.yaml --ckpt runs/myupdate_strat/best.pt
```

Nếu muốn cho script tự thử nhiều vòng hyperparameter:

```powershell
python scripts/auto_train_eval.py --config configs/logs_stft_train_strat_myupdate.yaml --max_rounds 5
```

Sau khi chạy xong, cần kiểm tra:

- `runs/.../best.pt`
- `runs/.../train_log.csv`
- `runs/.../train_curves.png`
- `runs/.../eval/report.txt`
- `runs/.../eval/confusion_matrix.png`
- `runs/.../eval/f1_per_class.png`

Bạn copy các artifact cần giữ về `figures/` khi đã chốt version dùng cho paper.

## 6. Bước 3: chạy temporal fine-tuning

Đây là bước paper đang coi là deployment-oriented evaluation.

Config temporal chuẩn hiện tại dùng:

- train: `[0,60]%`
- val: `[60,70]%`
- test: `[70,100]%`

Lệnh chạy:

```powershell
python train_logs.py --config configs/logs_stft_full_temporal_myupdate.yaml
python eval_logs.py --config configs/logs_stft_full_temporal_myupdate.yaml --ckpt runs/myupdate_temporal/best.pt
```

Khi chạy temporal, phải kiểm tra:

- `train.init_from` có trỏ đến checkpoint stratified đúng không
- `split_mode: temporal`
- `temporal_ttf` có đúng range cần test không

Sau eval, các file quan trọng là:

- `runs/.../eval/report.txt`
- `runs/.../eval/report_present.txt`
- `runs/.../eval/report_early_70_90.txt`
- `runs/.../eval/report_late_90_100.txt`
- `runs/.../eval/confusion_matrix.png`

Trong repo hiện tại, phần paper đang sử dụng các số cho:

- Early `70-90%`
- Late `90-100%`

Nên hai file `report_early_70_90.txt` và `report_late_90_100.txt` là nguồn số liệu trực tiếp cho phần này.

## 7. Bước 4: chạy full-range temporal nếu muốn cập nhật kết quả toàn vòng đời

Phần này map với đoạn trong `main.tex` đang viết:

- full lifecycle `[0,100]%`
- có báo cáo `majority vote`
- có báo cáo thêm `mean-logit`

Lệnh chạy:

```powershell
python train_logs.py --config configs/logs_stft_full_temporal_alltest_myupdate.yaml
python eval_logs.py --config configs/logs_stft_full_temporal_alltest_myupdate.yaml --ckpt runs/myupdate_temporal_alltest/best.pt
```

Cần kiểm tra:

- `temporal_ttf.test: [0.0, 100.1]`
- nếu bạn muốn số cho `vote`, nhớ chạy thêm:

```powershell
python eval_logs.py --config configs/logs_stft_full_temporal_alltest_myupdate.yaml --ckpt runs/myupdate_temporal_alltest/best.pt --agg vote
```

Lưu ý quan trọng:

- `train_logs.py` validate bằng `agg="mean"`
- `eval_logs.py` mặc định cũng là `agg="mean"`
- Paper hiện tại có một số con số full-range theo `majority vote`, nên nếu update paper bạn phải ghi rõ số nào đến từ `mean`, số nào đến từ `vote`

## 8. Bước 5: chạy baseline để đối sánh công bằng

Theo `BASELINE_COMPARISON_GUIDE.md`, tối thiểu cần có:

- `Classical vib (SVM, 8D)`
- `Vibration-only CNN`
- `Current best multi-modal`
- `Your new model`

Trong repo này:

- baseline deep vibration-only đã có config rõ ràng trong `configs/ablation/`
- baseline cổ điển phải được code và chạy trong `classical_baselines/`

Điều này quan trọng vì mục tiêu không chỉ là “có thêm một model classical”, mà là làm cho so sánh giữa:

- classical baseline
- deep vibration-only baseline
- current best multi-modal
- model mới của bạn

được thực hiện trên cùng một nền đánh giá.

### 8.1 Quy tắc khách quan cho classical baseline

Nếu paper update cần đối sánh với model cổ điển thì pipeline trong `classical_baselines/` phải đảm bảo:

- cùng `manifest`
- cùng split logic
- cùng TTF slice
- cùng file-wise metric

Trong repo hiện tại, phần này đã được chuẩn bị đúng hướng:

- `classical_baselines/pipeline.py` dùng lại logic chia dữ liệu của repo thông qua `LogsTTFDataset`
- config classical đang trỏ về cùng `data/manifest.csv`
- có cả config `stratified` và `temporal`
- đánh giá theo file-level, không phải window-level

Nói ngắn gọn:

- deep model và classical model có thể khác feature/model family
- nhưng phải cùng `samples`, cùng `splits`, cùng logic đánh giá

Đó mới là điều làm cho kết quả có tính khách quan khi đưa vào paper.

### 8.2 Nơi triển khai baseline cổ điển

Nếu bạn chuẩn bị so sánh model mới của mình với các model cổ điển, phần code nên được đặt trong:

- `classical_baselines/features.py`
- `classical_baselines/pipeline.py`
- `classical_baselines/configs/`

Không nên:

- viết tách rời ở một notebook khác
- dùng một manifest riêng cho classical
- tự chia train/test thủ công khác với deep pipeline

Nếu làm vậy thì kết quả so sánh sẽ không còn khách quan.

### 8.3 Cấu hình chung cần giữ đồng bộ

Khi thêm một classical baseline mới, cần giữ đồng bộ với deep pipeline ở các điểm sau:

- `manifest: data/manifest.csv`
- `split_mode: stratified` hoặc `split_mode: temporal`
- các mốc `temporal_ttf`
- logic train/val/test theo file
- đơn vị đánh giá cuối cùng là file-wise

Có thể khác:

- feature extractor
- model family như `SVM`, `LogReg`, `RandomForest`
- cách tổng hợp xác suất/logit của mô hình classical nếu vẫn quy về dự đoán file-wise nhất quán

### 8.4 Lệnh chạy baseline cổ điển

Ví dụ với baseline SVM 8D hiện có:

```powershell
python classical_baselines/train_classical.py --config classical_baselines/configs/svm_vib8_stratified.yaml
python classical_baselines/train_classical.py --config classical_baselines/configs/svm_vib8_temporal.yaml
```

Artifact đầu ra thường nằm ở:

- `runs/classical/.../model.pkl`
- `runs/classical/.../report_test.txt`
- `runs/classical/.../confusion_matrix_test.csv`
- `runs/classical/.../predictions_test.csv`

Khi đưa vào paper, bạn nên map các số classical này theo cùng cách map số cho deep models: stratified, temporal, và các slice nếu cần.

Nếu bạn đang làm một update nhỏ, tối thiểu hãy chạy lại:

- vibration-only stratified
- vibration-only temporal
- classical stratified
- classical temporal
- model mới stratified
- model mới temporal

Nếu bạn đang chốt paper, cần có bảng đối sánh đầy đủ hơn.

## 9. Bước 6: tổng hợp artifact vào `figures/`

Sau khi chốt run nào sẽ đưa vào paper, hãy copy artifact từ `runs/.../eval/` sang `figures/...` để tránh phụ thuộc vào một thư mục run tạm thời.

Đề xuất cấu trúc:

- `figures/stratified/`
- `figures/temporal/`
- `figures/temporal_alltest/`
- `figures/ablation/`

Cảnh báo quan trọng:

- `eval_logs.py` sinh file `f1_per_class.png`
- `main.tex` hiện đang gọi một số file tên `f1_present.png` và `f1_present_full.png`

Vì vậy, mỗi lần update hình bạn phải chọn 1 trong 2 cách:

1. Đổi tên file png trong `figures/` cho khớp với `main.tex`
2. Hoặc sửa đường dẫn trong `main.tex` để trỏ đúng tên file mới

Không nên để tình trạng tên file trong `main.tex` khác tên file thật trong repo.

## 10. Bước 7: rút số liệu đúng từ các report

Nguồn số liệu ưu tiên:

- Stratified 3-class: `report.txt`
- Temporal late-life: `report_early_70_90.txt`, `report_late_90_100.txt`
- Full-range: `report.txt`
- Present-class macro-F1: `report_present.txt`

Cách lấy số:

- Accuracy: lấy dòng `accuracy`
- Macro-F1:
  - nếu đủ 3 lớp, lấy dòng `macro avg`
  - nếu slice thiếu lớp, ưu tiên `report_present.txt` hoặc chỉ báo cáo F1 của lớp hiện diện
- F1 từng lớp: lấy dòng `healthy`, `degrading`, `fault`

Nguyên tắc viết paper:

- Nếu slice chỉ có 1 lớp, không nên viết Macro-F1 3 lớp như một kết quả "để so sánh"
- Nếu paper viết "present-class scoring", số liệu phải khớp logic đó

## 11. Bước 8: map kết quả vào `main.tex`

### 11.1 Abstract

Cập nhật:

- macro-F1 stratified
- điểm temporal quan trọng nhất
- kết quả full-lifecycle tốt nhất

Vị trí hiện tại: abstract ở đầu file `main.tex`.

Nếu model mới tốt hơn model cũ, đây là nơi cần sửa đầu tiên.

### 11.2 Introduction

Cập nhật bullet `Contributions` nếu thay đổi thực sự ở 1 trong 3 nhóm:

- phương pháp mới
- protocol mới
- kết quả mới

Nếu chỉ thay đổi nhỏ về hyperparameter, thường không cần đổi contribution.

### 11.3 Proposed Methodology

Sửa phần này nếu model mới thay đổi:

- temp feature
- fusion strategy
- STFT setting mô tả trong bài
- aggregation
- optimization workflow

Nếu code đã đổi mà text không đổi, paper sẽ sai.

### 11.4 Experimental Setup

Đây là phần cần sửa nếu config thay đổi:

- `window_seconds`, `hop_seconds`
- `n_fft`, `hop_length`
- `input_size`
- split ratio stratified
- temporal range
- LR / weight decay / epochs
- có hay không có `init_from`

Trong `main.tex`, bảng settings và đoạn split protocol phải khớp với config cuối cùng được dùng để sinh kết quả.

### 11.5 Results

Map như sau:

- `Stratified Evaluation (3-class)`:
  - lấy từ stratified `report.txt`
- `Temporal Evaluation (Deployment-Oriented)`:
  - lấy từ `report_early_70_90.txt` và `report_late_90_100.txt`
- `Full-Range Temporal (0--100%)`:
  - lấy từ alltest `report.txt`
  - nếu báo cáo `vote`, phải dùng output eval chạy với `--agg vote`

### 11.6 Ablation and Discussion

Chỉ đưa vào đây những ablation nào đã chạy lại thật sự.

Không được:

- giữ số cũ của model cũ
- nhưng text mới lại mô tả model mới

Nếu mở rộng thêm 1 branch/feature mới, tối thiểu nên có 1 ablation trả lời:

- bỏ thành phần mới đi thì điểm giảm bao nhiêu

## 12. Bước 9: cập nhật bảng và hình trong paper

### Bảng

Trong `main.tex` hiện có các bảng chính:

- `tab:stages`
- `tab:settings`
- `tab:abl_summary`

Khi sửa:

- `tab:settings` phải khớp config train thật sự
- `tab:abl_summary` phải khớp các run đã chốt
- nếu thêm variant mới, thêm dòng mới
- nếu bỏ variant cũ, xóa dòng cũ

### Hình

Trong `main.tex` hiện có:

- `confusion_matrix.png`
- `f1_present.png`
- `confusion_matrix_full.png`
- `f1_present_full.png`

Cần đảm bảo 2 việc:

1. File ảnh tồn tại thật trong thư mục build LaTeX
2. Nội dung ảnh khớp với kết quả đang được mô tả trong text

Một lỗi rất dễ gặp:

- text nói về kết quả mới
- nhưng hình vẫn là hình của lần chạy cũ

Mỗi lần update paper, nên tick checklist:

- đã thay hình stratified chưa
- đã thay hình temporal chưa
- đã thay hình full-range chưa
- đã thay hình ablation chưa

## 13. Bước 10: checklist trước khi sửa LaTeX

Hãy chỉ sửa `main.tex` sau khi đã có đủ 5 nhóm dữ liệu:

1. Kết quả stratified của model mới
2. Kết quả temporal của model mới
3. Kết quả full-range nếu paper có báo cáo mục này
4. Kết quả baseline cần thiết để đối sánh
5. Hình và bảng đã chốt tên file

Nếu chưa đủ, rất dễ sửa paper nửa chừng nửa dữ liệu.

## 14. Bước 11: checklist khi sửa `main.tex`

Sau khi sửa, tự check lại các điểm sau:

- Abstract có số nào khác phần Results không?
- Introduction contribution có nói quá mức so với kết quả thật không?
- Methodology có mô tả đúng model mới không?
- Experimental Setup có khớp config mới không?
- Results có khớp report mới không?
- Ablation table có khớp số trong hình/báo cáo không?
- Conclusion có lặp lại đúng kết quả tốt nhất không?

Nếu có 1 chỗ nào không khớp, ưu tiên sửa cho text và bảng/hình đồng bộ.

## 15. Bước 12: quy trình update đề xuất cho mỗi lần nâng cấp model

Thứ tự thực tế nên làm:

1. Tạo config mới cho stratified
2. Chạy stratified, chốt checkpoint tốt nhất
3. Tạo config temporal fine-tune từ checkpoint đó
4. Chạy temporal
5. Chạy alltest nếu paper cần full lifecycle
6. Chạy baseline đối sánh cần thiết
7. Copy artifact cuối cùng vào `figures/`
8. Rút số từ report
9. Sửa `main.tex`
10. Build PDF và đọc lại toàn bộ paper

Thứ tự này giúp tránh việc sửa paper quá sớm.

## 16. Mẫu bảng ghi chép cho mỗi bản update

Nên tự tạo 1 bảng note tạm mỗi lần update:

```text
Model name:
Config stratified:
Config temporal:
Config alltest:
Checkpoint stratified:
Temporal init_from:
Best stratified macro-F1:
Early 70-90 accuracy / F1:
Late 90-100 accuracy / F1:
Full-range mean accuracy / macro-F1:
Full-range vote accuracy / macro-F1:
Baselines compared:
Figures copied:
Sections in main.tex to update:
```

Chỉ cần điền đầy đủ form này là việc sửa paper sẽ nhanh hơn rất nhiều.

## 17. Lưu ý đặc biệt cho repo này

- `BASELINE_COMPARISON_GUIDE.md` nói rõ rằng: đối sánh công bằng quan trọng hơn "điểm cao"
- `classical_baselines/` là nơi chuẩn để cài và chạy các baseline cổ điển phục vụ so sánh khách quan
- `main.tex` hiện đang mô tả rất cụ thể về protocol, nên nếu config đổi thì text phải đổi theo
- `eval_logs.py` sinh nhiều report con theo TTF slice; nên ưu tiên lấy số từ đúng file thay vì nhớ bằng tay
- `scripts/aggregate_ablation.py` hữu ích khi bạn có nhiều run theo seed và muốn tổng hợp mean/std
- nếu dùng `auto_train_eval.py`, nhớ ghi rõ paper đang báo cáo kết quả của round nào, config nào

## 18. Kết luận ngắn

Quy trình đúng cho repo này là:

- chốt code và config trước
- chạy stratified -> temporal -> alltest
- đối sánh với baseline bắt buộc
- copy artifact cuối cùng vào `figures/`
- rồi mới cập nhật `main.tex`

Nếu bạn muốn, bước tiếp theo tôi có thể tạo thêm:

- một checklist cực ngắn 1 trang để dùng mỗi lần update
- hoặc một template `paper_update_log.md` để bạn điền kết quả cho từng run
