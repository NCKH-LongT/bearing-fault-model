# Hướng dẫn chạy code phục vụ bài báo

Tài liệu này mô tả pipeline đang tạo dữ liệu, checkpoint, báo cáo, figure và PDF cho bài báo **Three-Stage Bearing Health Classification Using Vibration Short-Time Fourier Transform and Temperature Trends in a Run-to-Failure Setting**.

Để chạy quy trình revision mới theo từng bước, xem `REVISION_STEP_BY_STEP.md` và dùng `run_revision.py` trong cùng thư mục.

Mọi lệnh bên dưới được chạy từ thư mục gốc repository, trừ phần biên dịch LaTeX.

## 1. Pipeline hiện tại

```text
data/*.csv + data/manifest.csv
        │
        ├── Stratified pretraining (checkpoint chuẩn auto_r22)
        │
        └── Temporal fine-tuning [0,60] / [60,70] / [70,100]% TTF
                    │
                    ├── Temporal evaluation (mean-logit)
                    ├── Whole-trajectory evaluation (mean-logit)
                    └── Whole-trajectory evaluation (majority vote)
                                      │
                                      └── paper/figures/
                                                │
                                                └── paper/main.pdf
```

Các entry point chính:

- `train_logs.py`: huấn luyện mô hình deep.
- `eval_logs.py`: đánh giá theo file và sinh report/figure.
- `scripts/run_paper_sync.py`: điều phối temporal train, các lần eval và đồng bộ figure.
- `scripts/run_comparison_baseline.py`: chạy các baseline đã đăng ký.
- `paper/main.tex`: mã nguồn bài báo.

## 2. Môi trường

Khuyến nghị Python 3.10 trở lên. Tạo virtual environment trên Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Pipeline cần tối thiểu:

- PyTorch
- NumPy
- PyYAML
- scikit-learn
- Matplotlib

Cần cài bản PyTorch phù hợp với CPU/CUDA của máy trước, sau đó cài các gói còn lại:

```bash
python -m pip install -r requirements-paper.txt
```

Kiểm tra môi trường:

```bash
python -c "import torch, numpy, yaml, sklearn, matplotlib; print(torch.__version__); print(torch.cuda.is_available())"
```

Trước khi train dài, chạy smoke test bằng một sample thật:

```bash
python paper/run_revision.py smoke
```

Để tạo PDF cần có `latexmk`, `pdflatex` và `bibtex`:

```bash
latexmk -v
pdflatex --version
bibtex --version
```

## 3. Dữ liệu đầu vào

File bắt buộc:

```text
data/manifest.csv
data/LogFile_*.csv
```

Mỗi CSV tín hiệu phải có ít nhất bốn cột theo thứ tự:

```text
vib_x, vib_y, temp_bearing, temp_atm
```

Manifest phải có các cột:

```text
file, run_id, ttf_percent, fault_type
```

Nhãn hợp lệ là `healthy`, `degrading`, `fault`. Kiểm tra nhanh manifest:

```bash
python scripts/check_manifest_ttf.py configs/best_temporal.yaml
python scripts/peek_manifest.py
```

Workspace hiện có 129 file tín hiệu cho một run:

- Healthy: 77 file.
- Degrading: 39 file.
- Fault: 13 file.
- Temporal test `[70,100]%`: 26 Degrading và 13 Fault; không có Healthy.

## 4. Tiền xử lý và mô hình

Mỗi file được cắt thành cửa sổ 1 giây ở 25.6 kHz, bước nhảy 0.5 giây.

Nhánh rung:

1. Lấy hai trục rung.
2. Thay NaN/Inf và z-score theo từng cửa sổ/kênh.
3. STFT với `n_fft=2048`, `hop_length=512`, cửa sổ Hann.
4. Lấy log-magnitude và chuẩn hóa theo tần số qua trục thời gian.
5. Resize thành tensor hai kênh `2 × 160 × 160`.
6. Đưa qua ResNet2D nhỏ để tạo embedding 256 chiều.

Nhánh nhiệt độ tạo sáu đặc trưng: mean, standard deviation và slope cho nhiệt độ ổ bi và môi trường. Sáu giá trị được chiếu thành embedding 32 chiều. Hai embedding được nối lại và đưa qua linear classifier ba lớp.

Trong đánh giá, dự đoán mặc định của một file là `argmax` của trung bình pre-softmax logits trên toàn bộ cửa sổ. Tùy chọn `--agg vote` dùng majority vote từ dự đoán từng cửa sổ.

## 5. Các config chuẩn

```text
configs/best_stratified_ref.yaml
configs/best_temporal.yaml
configs/best_fullrange_eval.yaml
```

- `best_stratified_ref.yaml` chỉ mô tả checkpoint lịch sử `auto_r22`; không bảo đảm tái tạo checkpoint đó từ đầu.
- `best_temporal.yaml` fine-tune từ `runs/logs_stft_strat/auto_r22/best.pt`.
- `best_fullrange_eval.yaml` đánh giá checkpoint temporal trên toàn bộ `[0,100]%` TTF.

## 6. Điều kiện còn thiếu trong workspace hiện tại

Pipeline chuẩn cần file:

```text
runs/logs_stft_strat/auto_r22/best.pt
```

Checkpoint này hiện không có trong workspace. Vì vậy chưa thể tái tạo đúng kết quả headline của paper. `scripts/run_paper_sync.py` sẽ dừng với thông báo rõ ràng thay vì tiếp tục fine-tune bằng trọng số ngẫu nhiên.

Sau khi khôi phục checkpoint, kiểm tra:

```bash
test -f runs/logs_stft_strat/auto_r22/best.pt && echo OK
```

## 7. Chạy pipeline bài báo

### 7.1 Chạy tự động

Sau khi môi trường, dữ liệu và checkpoint đã đầy đủ:

```bash
python scripts/run_paper_sync.py --sync-figures
```

Lệnh này thực hiện:

1. Fine-tune temporal bằng `configs/best_temporal.yaml`.
2. Đánh giá temporal bằng mean-logit.
3. Đánh giá toàn trajectory bằng mean-logit.
4. Đánh giá toàn trajectory bằng majority vote.
5. Copy report và figure được chọn vào `paper/figures/`.

### 7.2 Chạy thủ công

Fine-tune temporal:

```bash
python train_logs.py --config configs/best_temporal.yaml
```

Checkpoint đầu ra:

```text
runs/paper_sync/temporal/best.pt
```

Đánh giá temporal `[70,100]%` bằng mean-logit:

```bash
python eval_logs.py \
  --config configs/best_temporal.yaml \
  --ckpt runs/paper_sync/temporal/best.pt
```

Đánh giá toàn trajectory bằng mean-logit và vote:

```bash
python eval_logs.py \
  --config configs/best_fullrange_eval.yaml \
  --ckpt runs/paper_sync/temporal/best.pt

python eval_logs.py \
  --config configs/best_fullrange_eval.yaml \
  --ckpt runs/paper_sync/temporal/best.pt \
  --agg vote
```

Nếu các run đã tồn tại và chỉ cần đồng bộ figure:

```bash
python scripts/run_paper_sync.py --skip-train --sync-figures
```

## 8. Artifact đầu ra

```text
runs/paper_sync/temporal/best.pt
runs/paper_sync/temporal/train_log.csv
runs/paper_sync/temporal/eval/
runs/paper_sync/fullrange/eval/
runs/paper_sync/fullrange/eval_vote/
paper/figures/stratified/
paper/figures/temporal/
paper/figures/fullrange/
```

Mỗi thư mục `eval*` có thể chứa:

- `report.txt`: report ba lớp cố định.
- `report_present.txt`: report chỉ trên lớp thật sự xuất hiện trong lát đánh giá.
- `confusion_matrix.csv/png`.
- `f1_per_class.png`, `f1_present.png`.
- Report/confusion matrix cho lát 70–90% và 90–100%.

## 9. Chạy baseline

Liệt kê baseline:

```bash
python scripts/run_comparison_baseline.py --list
```

SVM 8-D:

```bash
python scripts/run_comparison_baseline.py \
  --baseline svm_vib8 --protocol stratified --action train_eval

python scripts/run_comparison_baseline.py \
  --baseline svm_vib8 --protocol temporal --action train_eval
```

Vibration-only CNN phải chạy stratified trước temporal:

```bash
python scripts/run_comparison_baseline.py \
  --baseline vibration_only_cnn --protocol stratified --action train_eval

python scripts/run_comparison_baseline.py \
  --baseline vibration_only_cnn --protocol temporal --action train_eval
```

Không dùng protocol `svm_vib8/temporal_pure` làm baseline chính, vì temporal train không có ví dụ Fault.

## 10. Biên dịch bài báo

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c main.tex
cd ..
```

Kết quả:

```text
paper/main.pdf
```

`latexmk -c` dọn file phụ nhưng giữ PDF và `main.bbl`.

## 11. Kiểm tra số liệu trước khi cập nhật paper

Không copy số trực tiếp từ terminal. Đọc các report đã lưu và đối chiếu support, confusion matrix:

```bash
sed -n '1,30p' paper/figures/stratified/report.txt
sed -n '1,30p' paper/figures/temporal/report.txt
sed -n '1,30p' paper/figures/temporal/report_present.txt
sed -n '1,30p' paper/figures/fullrange/report.txt
```

Các giá trị đang được bản thảo sử dụng:

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Stratified | 0.7241 | 0.7762 |
| Temporal `[70,100]%` | 0.8974 | 0.9044 trên present classes |
| Whole trajectory `[0,100]%`, vote | 0.8915 | 0.8739 |

Các số này là artifact lịch sử. Chỉ cập nhật paper sau khi run mới tái tạo được report tương ứng.

## 12. Cảnh báo về tính hợp lệ của protocol

Đây là phần phải xử lý trước khi dùng kết quả để trả lời reviewer.

### 12.1 Stratified pretraining có giao với temporal test

Stratified split lấy mẫu file ngẫu nhiên từ toàn bộ trajectory. Toàn bộ 13 file Fault nằm trong vùng TTF 90–100%, đồng thời stratified training lấy khoảng 60% file của từng lớp. Theo split logic hiện tại, checkpoint stratified chắc chắn đã học một phần file Fault thuộc vùng temporal test trước khi temporal fine-tuning.

Do đó không nên mô tả pipeline hiện tại là hoàn toàn leakage-free hoặc là kiểm tra generalization độc lập. Cách gọi phù hợp hơn là:

```text
within-run forward-transfer evaluation
```

hoặc:

```text
chronology-oriented late-life analysis with stratified initialization
```

### 12.2 Temporal train không phải bài toán ba lớp độc lập

Với nhãn hiện tại, train `[0,60]%` gần như chỉ có Healthy; validation `[60,70]%` chỉ có Degrading; test `[70,100]%` có Degrading và Fault. Phase temporal chỉ hoạt động nhờ checkpoint stratified đã nhìn thấy đủ lớp.

### 12.3 Full-range không phải held-out test

`configs/best_fullrange_eval.yaml` đặt test range thành `[0,100.1]`, bao gồm cả vùng dùng trong train và validation. Vì vậy phải gọi kết quả này là:

```text
whole-trajectory retrospective evaluation
```

Không gọi là full-lifecycle held-out test và không nên dùng làm kết quả generalization chính.

### 12.4 Các lát 70–90% và 90–100% là single-class

- 70–90% chỉ chứa Degrading.
- 90–100% chỉ chứa Fault.

F1 cao ở các lát này chỉ phản ánh khả năng nhận đúng lớp hiện diện; không chứng minh phân biệt ba lớp. Chuyển chúng xuống phần secondary/transition analysis.

## 13. Hướng giải quyết revision theo thứ tự ưu tiên

Phần này chuyển các đề xuất trong `bearing_paper_revision_guide.md` thành hành động cụ thể cho repository hiện tại.

### P0 — Audit và sửa protocol trước khi chạy thêm model

1. Sinh danh sách file cố định cho từng phase/split.
2. Báo cáo số file, số cửa sổ và support từng lớp.
3. Tính giao file giữa stratified train và temporal test.
4. Ghi rõ toàn bộ layer được fine-tune; code hiện không freeze layer nào.
5. Đổi tên full-range thành whole-trajectory retrospective evaluation.
6. Bỏ claim “leakage-aware generalization” nếu vẫn dùng `auto_r22`.

Giải pháp mạnh nhất là tạo protocol group-disjoint hoặc run-disjoint với nhiều run/bearing. Nếu chỉ có một run, cần hạ claim và không xem temporal result hiện tại là bằng chứng cross-bearing generalization.

### P1 — Xây dựng primary evaluation hợp lệ

Primary result nên là một tập held-out nhiều lớp, không có file xuất hiện trong pretraining. Có ba lựa chọn theo mức độ mạnh:

1. Leave-one-run/bearing-out nếu lấy thêm được run hoặc bearing.
2. Group-disjoint split theo bearing/condition.
3. Nếu chỉ còn một run: giữ một tập file nhiều lớp hoàn toàn ngoài pretraining và mô tả đây là within-run held-out evaluation, không phải cross-bearing generalization.

Không dùng full-range hoặc single-class slice làm primary result.

Với đúng một run và nhãn được xác định trực tiếp theo TTF, không thể đồng thời có một phép thử vừa chronology-forward, vừa chứa đủ ba lớp, vừa cho train nhìn thấy lớp Fault. Phương án trung thực trước mắt là dùng file-wise stratified group-disjoint test làm primary multi-class result và xem các lát temporal là secondary trajectory analysis. Muốn có primary chronology-forward ba lớp thực sự cần thêm run/bearing để train nhìn thấy các lớp trên run khác rồi test trên run được giữ lại.

### P2 — Baseline và ablation tối thiểu

Ưu tiên bộ đủ trả lời reviewer nhưng không mở rộng quá nhanh:

1. SVM vibration 8-D.
2. Vibration-only CNN cùng backbone/STFT.
3. Temperature-only.
4. Simple late concatenation hiện tại.
5. Gated fusion đã có khung ở `models/gated_fusion.py` sau khi tích hợp đầy đủ vào train/eval.
6. Một baseline temporal mạnh như 1D CNN/TCN hoặc CNN-GRU.

Mọi baseline phải dùng cùng danh sách file, seed, aggregation và metric.

### P3 — Efficiency và robustness

Thêm script đo:

- số tham số và kích thước checkpoint;
- latency STFT, model và end-to-end ở batch size 1;
- throughput và peak GPU memory;
- CPU latency;
- FLOPs nếu thêm dependency phù hợp.

Robustness tối thiểu:

- vibration Gaussian noise ở 20/10/5/0 dB;
- temperature missing 10/30/50%;
- temperature drift và stuck-at-constant;
- mất một trục vibration;
- so sánh zero-fill, modality dropout và fallback vibration-only.

### P4 — Độ tin cậy thống kê

Chạy tối thiểu năm seed trên protocol đã sửa, báo cáo mean ± standard deviation. Tính bootstrap 95% CI và paired comparison ở cấp file, không bootstrap các cửa sổ tương quan trong cùng file.

### P5 — Sửa bài theo evidence

- Abstract: bỏ headline full-range nếu nó chứa train data.
- Contributions: mô tả kiến trúc hiện tại là compact standard late fusion, không claim fusion architecture mới.
- Results: primary held-out multi-class → baseline → robustness → efficiency → retrospective trajectory.
- Discussion/Limitations: ghi rõ single run, threshold thực nghiệm và khả năng residual temporal correlation.
- Conclusion: giới hạn claim ở within-run performance cho đến khi có cross-run/cross-bearing evaluation.

## 14. Checklist chạy một revision hợp lệ

- [ ] Đã khôi phục hoặc thay thế có kiểm soát checkpoint stratified.
- [ ] Đã lưu danh sách file cố định cho mọi split.
- [ ] Pretraining train và primary test không giao file.
- [ ] Primary test chứa nhiều lớp.
- [ ] Mọi baseline dùng cùng split và aggregation.
- [ ] Chạy ít nhất năm seed.
- [ ] Có support và confidence interval ở cấp file.
- [ ] Full-range được ghi là retrospective.
- [ ] Single-class slices chỉ là secondary analysis.
- [ ] Có efficiency và robustness report.
- [ ] `paper/main.tex` chỉ được cập nhật từ artifact đã lưu.
- [ ] `latexmk` biên dịch thành công.

## 15. Troubleshooting

### Thiếu checkpoint `auto_r22`

Khôi phục đúng checkpoint lịch sử nếu mục tiêu là tái tạo paper hiện tại. Nếu không thể khôi phục, cần định nghĩa một protocol/checkpoint mới, chạy lại toàn bộ baseline và cập nhật mọi con số; không dùng trọng số ngẫu nhiên để thay thế mà vẫn giữ số cũ.

### Hết GPU memory

Giảm `train.batch_size`, không thay `input_size` hoặc STFT nếu mục tiêu là so sánh trực tiếp với artifact cũ. Ghi lại mọi thay đổi config.

### `ModuleNotFoundError`

Đảm bảo virtual environment đã activate và chạy script bằng chính interpreter đã cài dependency:

```bash
which python
python -c "import torch, numpy, yaml, sklearn"
```

### Report không khớp paper

Kiểm tra checkpoint, config, aggregation (`mean` hay `vote`), seed, danh sách file và `val_max_windows`. Không sửa thủ công report hoặc figure.
