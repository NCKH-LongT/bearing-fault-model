# Quy trình chạy lại và revision bài báo từng bước

Tài liệu này sử dụng runner:

```text
paper/run_revision.py
```

Runner được xây dựng từ các ưu tiên trong `paper/bearing_paper_revision_guide.md`. Mục tiêu là tạo một primary evaluation file-wise, nhiều lớp và không giao file giữa train/validation/test; các kết quả temporal/full-trajectory cũ được giữ ở vai trò secondary analysis.

## Nguyên tắc

- Artifact revision mới được ghi vào `runs/revision/`.
- Report dùng để sửa bài được copy vào `paper/revision_artifacts/`.
- Runner không tự sửa số trong `main.tex`.
- Runner không ghi đè `paper/figures/` trừ khi chủ động chạy bước `secondary`.
- Bước `secondary` cần checkpoint lịch sử `auto_r22` và không được xem là primary generalization test.

## Bước 0 — Chuẩn bị môi trường

Từ thư mục gốc repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Cài dependency thông thường của paper trước:

```bash
python -m pip install -r requirements-paper.txt
```

### Cài PyTorch

Máy hiện tại đang dùng Python 3.14.4. PyTorch 2.11 hỗ trợ Python 3.14, nhưng NVIDIA driver hiện chưa hoạt động (`nvidia-smi` không giao tiếp được với driver).

Có hai lựa chọn.

#### Lựa chọn A — Cài CPU để chạy preflight, smoke và audit ngay

```bash
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Kiểm tra:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Kết quả `False` cho CUDA là đúng với bản CPU. Có thể tiếp tục đến `smoke` và `audit`, nhưng không nên chạy train 80 epoch trên CPU vì sẽ rất chậm.

#### Lựa chọn B — Sửa NVIDIA driver rồi cài CUDA để train

Trước hết kiểm tra ngoài virtual environment:

```bash
nvidia-smi
```

Chỉ tiếp tục cài PyTorch CUDA khi lệnh này hiển thị được GPU và driver version. Nếu vẫn lỗi, xem driver Ubuntu đề xuất và cài bằng `ubuntu-drivers`:

```bash
sudo ubuntu-drivers list
sudo ubuntu-drivers install
sudo reboot
```

Sau khi reboot, chạy lại `nvidia-smi`. Nếu máy bật Secure Boot, quá trình cài có thể yêu cầu tạo/enroll khóa MOK; cần hoàn tất bước này khi reboot để kernel nạp NVIDIA module.

Khi driver đã hoạt động, cài wheel CUDA 12.8:

```bash
python -m pip uninstall -y torch
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra GPU từ PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Chỉ chạy `primary`, `baselines` hoặc `all` khi kết quả là `True` và tên GPU được in ra.

Lệnh cài đặt dựa trên hướng dẫn chính thức:

- <https://pytorch.org/get-started/locally/>
- <https://pytorch.org/get-started/previous-versions/>
- <https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/>

Xác nhận mọi lệnh đang dùng Python trong `.venv`:

```bash
which python
python -c "import torch, numpy, yaml, sklearn, matplotlib; print(torch.__version__); print(torch.cuda.is_available())"
```

Kiểm tra runner:

```bash
python paper/run_revision.py --help
```

## Bước 1 — Preflight

```bash
python paper/run_revision.py preflight
```

Bước này kiểm tra:

- `data/manifest.csv` và toàn bộ file tín hiệu được tham chiếu.
- Ba config revision.
- Python packages cần thiết.
- `latexmk`, `pdflatex`, `bibtex`.
- Phân bố lớp trong manifest.
- Trạng thái checkpoint lịch sử `auto_r22`.

Checkpoint `auto_r22` không cần cho primary revision mới. Nó chỉ cần nếu tái tạo temporal result cũ.

Nếu preflight báo thiếu module dù đã cài, nguyên nhân thường là lệnh đang dùng `/usr/bin/python3` thay vì `.venv/bin/python`.

Sau khi cài PyTorch, chạy lại:

```bash
python paper/run_revision.py preflight
```

Không chuyển sang bước sau cho đến khi preflight kết thúc bằng thông báo `Training inputs OK` và `LaTeX tools OK`.

## Bước 2 — Smoke test code và dữ liệu thật

```bash
python paper/run_revision.py smoke
```

Bước này không train model. Nó:

1. Đọc hai giây đầu của một file CSV thật từ primary train split.
2. Tạo STFT tensor `2×160×160`.
3. Tạo temperature descriptor 6-D.
4. Chạy forward qua mô hình multimodal.
5. Tính cross-entropy và chạy backward một batch trên CPU.

Kết quả mong đợi:

```text
[smoke] OK: x=(2, 160, 160), temp=(6,), logits=(1, 3), loss=...
```

Nếu bước này lỗi, chưa chạy `primary`. Sửa lỗi dữ liệu, shape hoặc dependency trước.

## Bước 3 — Audit protocol

```bash
python paper/run_revision.py audit
```

Kết quả:

```text
paper/revision_artifacts/protocol_audit.md
```

Báo cáo gồm:

- Số file và lớp của primary stratified split.
- Số file và lớp của temporal split cũ.
- Giao file giữa train/validation/test.
- Giao giữa stratified pretraining và temporal test.
- Diễn giải vì sao temporal/full-range cũ không phải primary generalization result.

Đọc audit trước khi chạy huấn luyện:

```bash
sed -n '1,240p' paper/revision_artifacts/protocol_audit.md
```

## Bước 4 — Xem trước lệnh tốn thời gian

```bash
python paper/run_revision.py primary --dry-run
python paper/run_revision.py baselines --dry-run
python paper/run_revision.py all --dry-run
```

`--dry-run` không train hoặc ghi artifact thí nghiệm.

## Bước 5 — Chạy primary multimodal

```bash
python paper/run_revision.py primary
```

Config:

```text
configs/revision_primary_multimodal.yaml
```

Thiết lập chính:

- File-wise stratified `0.6/0.2/0.2`.
- Seed 42.
- STFT `2048/512`, ảnh `160×160`.
- Hai trục vibration + temperature descriptor 6-D.
- Train từ đầu, không dùng `auto_r22`.
- Test chứa Healthy, Degrading và Fault.

Đầu ra:

```text
runs/revision/primary_multimodal/best.pt
runs/revision/primary_multimodal/eval/
paper/revision_artifacts/primary_multimodal/
```

Đọc kết quả:

```bash
sed -n '1,40p' paper/revision_artifacts/primary_multimodal/report.txt
```

## Bước 6 — Chạy baseline cùng protocol

```bash
python paper/run_revision.py baselines
```

Runner chạy:

1. Vibration-only CNN với cùng split, STFT, optimizer và seed.
2. SVM vibration handcrafted 8-D với cùng file split.

Config:

```text
configs/revision_primary_vibration_only.yaml
classical_baselines/configs/revision_svm_vib8_stratified.yaml
```

Đầu ra:

```text
paper/revision_artifacts/primary_vibration_only/
paper/revision_artifacts/svm_vib8/
```

Kiểm tra:

```bash
sed -n '1,40p' paper/revision_artifacts/primary_vibration_only/report.txt
sed -n '1,40p' paper/revision_artifacts/svm_vib8/report_test.txt
```

Không so sánh Accuracy đơn lẻ. Cần lấy ít nhất Accuracy, Macro-F1 và F1 từng lớp trên cùng test support.

## Bước 7 — Đánh giá evidence trước khi sửa bài

Tạo bảng làm việc từ ba report:

| Model | Accuracy | Macro-F1 | Healthy F1 | Degrading F1 | Fault F1 |
|---|---:|---:|---:|---:|---:|
| SVM vibration 8-D | | | | | |
| Vibration-only CNN | | | | | |
| Multimodal late fusion | | | | | |

Chỉ cập nhật paper nếu:

- Ba model dùng đúng cùng file split.
- Test support giống nhau.
- Không dùng số từ artifact cũ để lấp ô của run mới.
- Report và confusion matrix đã được lưu.

## Bước 8 — Temporal/full-trajectory cũ, nếu cần

Chỉ chạy bước này để tái tạo secondary analysis cũ:

```bash
python paper/run_revision.py secondary
```

Yêu cầu:

```text
runs/logs_stft_strat/auto_r22/best.pt
```

Bước này gọi `scripts/run_paper_sync.py --sync-figures` và có thể cập nhật `paper/figures/temporal` cùng `paper/figures/fullrange`.

Trong bản revision:

- Gọi `[0,100]%` là “whole-trajectory retrospective evaluation”.
- Không gọi đây là held-out full-lifecycle test.
- Không dùng lát single-class 70–90% hoặc 90–100% làm primary result.
- Không claim temporal result là leakage-free nếu khởi tạo từ stratified checkpoint có giao file với temporal test.

## Bước 9 — Sửa `main.tex`

Sửa bài theo thứ tự:

1. Abstract: dùng primary multi-class held-out result; bỏ full-range khỏi headline nếu chứa train data.
2. Contributions: mô tả phương pháp hiện tại là compact late-fusion pipeline, không claim fusion architecture hoàn toàn mới.
3. Experimental Setup: thêm bảng file/class distribution lấy từ `protocol_audit.md`.
4. Results: primary evaluation → baseline → secondary trajectory analysis.
5. Discussion: phân tích Degrading/Fault và giới hạn single-run.
6. Limitations: ghi rõ chưa chứng minh cross-run/cross-bearing generalization.
7. Conclusion: giới hạn claim ở within-run file-level classification.

Không tự động copy số vào LaTeX. Việc này cần đọc report và xác nhận protocol trước.

## Bước 10 — Biên dịch LaTeX

```bash
python paper/run_revision.py latex
```

Hoặc thủ công:

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

## Bước 11 — Chạy chuỗi primary hoàn chỉnh

Sau khi đã kiểm tra dry-run:

```bash
python paper/run_revision.py all
```

`all` chạy:

1. Preflight.
2. Smoke test bằng dữ liệu thật.
3. Protocol audit.
4. Primary multimodal.
5. Vibration-only và SVM baselines.
6. Biên dịch LaTeX.

`all` cố ý không chạy `secondary`, vì secondary dùng protocol lịch sử có hạn chế leakage và thiếu checkpoint trong workspace hiện tại.

## Bước 12 — Khi nào cần sửa code

Code hiện đã được bổ sung để:

- fail-fast khi thiếu dependency, dữ liệu hoặc checkpoint;
- tách primary revision khỏi temporal artifact lịch sử;
- chạy smoke test shape/forward/backward;
- ghi run mới vào `runs/revision/`;
- không tự động dùng số cũ cho bài revision.

Chỉ sửa code model/train/eval khi smoke test hoặc artifact cho thấy vấn đề cụ thể. Sau mỗi sửa đổi:

```bash
python -m py_compile train_logs.py eval_logs.py paper/run_revision.py
python paper/run_revision.py smoke
python paper/run_revision.py audit
```

Nếu thay kiến trúc, preprocessing, split hoặc metric:

1. Tạo config mới, không sửa artifact lịch sử.
2. Đổi `log.out_dir` để không ghi đè run cũ.
3. Chạy lại cả multimodal và baseline liên quan.
4. Ghi thay đổi vào Method/Experimental Setup.
5. Chỉ cập nhật `main.tex` từ report mới.

## Bước 13 — Những phần reviewer yêu cầu nhưng runner chưa tự động hóa

Runner hiện tạo minimum defensible revision workflow, chưa hoàn thành toàn bộ revision guide. Các phần tiếp theo cần triển khai riêng:

- Temperature-only baseline.
- Gated fusion được tích hợp thật vào model train/eval.
- 1D CNN, TCN hoặc CNN-GRU baseline.
- Năm seed và mean ± standard deviation.
- Bootstrap 95% CI ở cấp file.
- Efficiency: params, FLOPs, CPU/GPU latency và memory.
- Robustness: vibration noise, temperature drift/missing và mất vibration axis.
- Cross-run/cross-bearing experiment.

Không nên tuyên bố đã giải quyết các reviewer comment này trước khi có code và artifact tương ứng.

## Checklist cuối

- [ ] `preflight` thành công.
- [ ] `smoke` chạy forward/backward thành công.
- [ ] Đã đọc `protocol_audit.md`.
- [ ] Primary test file-disjoint và có ba lớp.
- [ ] Baseline dùng cùng split.
- [ ] Đã đối chiếu support trong report.
- [ ] Full-range được gọi là retrospective.
- [ ] Single-class slices chỉ là secondary.
- [ ] Claim generalization đã được hạ đúng evidence.
- [ ] `main.tex` dùng số từ artifact revision.
- [ ] `latex` thành công.
