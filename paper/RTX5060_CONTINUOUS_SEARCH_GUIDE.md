# Chạy liên tục trên RTX 5060 Ti 16 GB để revision bài báo

Guide này dùng toàn bộ pipeline revision theo nguyên tắc: **tìm hyperparameter trên validation, khóa test cho đến bước xác nhận cuối**. Không chọn trial dựa trên test vì việc đó làm kết quả bài báo bị test-set overfitting.

## 1. Trạng thái máy đã xác nhận

```text
GPU: NVIDIA GeForce RTX 5060 Ti
VRAM: 15.48 GiB usable
Driver: 595.84
PyTorch: 2.11.0+cu128
CUDA available: True
Compute capability: 12.0
```

Một run `paper/run_revision.py primary` cũ đang chạy khi guide được tạo. Không chạy queue mới song song. Kiểm tra:

```bash
pgrep -af 'train_logs.py|eval_logs.py|paper/run_revision.py primary'
```

Chỉ tiếp tục khi lệnh không còn trả về tiến trình train/eval. Nếu run được mở trong terminal khác, nên để nó hoàn thành tự nhiên.

## 2. Vì sao GPU trước đây không chạy hết

Đo thực tế cho thấy GPU chỉ khoảng 0–21% utilization và dùng gần 1.4 GB VRAM. Các nguyên nhân:

- 130 CSV tổng cộng khoảng 17.26 GB, trung bình 136 MB/file.
- CSV được parse lại trong DataLoader workers.
- Worker cache làm RAM 30 GB gần đầy và sử dụng swap.
- Dataset cũ chỉ tạo khoảng một training sample/file/epoch, tức khoảng 71 sample train mỗi epoch.
- Batch size cũ là 24 nên model nhỏ không đủ tải RTX 5060 Ti.

Code revision đã được sửa:

- cache CSV sang `.npy` float32 và đọc memory-map;
- `samples_per_file` tạo nhiều cửa sổ ngẫu nhiên cho mỗi file trong một epoch;
- batch search 64/96/128;
- 8 DataLoader workers;
- AMP, TF32 và channels-last;
- test không được gọi trong hyperparameter search.
- phân bổ Fault revision thành 7 train / 3 validation / 3 test thay vì cấu hình cũ 3/5/5; test nhỏ nên vẫn phải báo độ bất định và limitation.

GPU không nhất thiết luôn hiện 100% vì còn thời gian validation và chuẩn bị dữ liệu. Mục tiêu là utilization cao ổn định trong training, không phải lấp đầy VRAM bằng mọi giá.

## 3. Kiểm tra môi trường

```bash
source .venv/bin/activate

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))"
nvidia-smi
python paper/run_revision.py preflight
```

Phải thấy CUDA `True` và tên RTX 5060 Ti.

## 4. Tạo cache dữ liệu một lần

Không chạy bước này đồng thời với run cũ vì cả hai cùng dùng nhiều CPU/RAM.

```bash
python scripts/cache_dataset_npy.py --workers 2
```

Cache nằm tại:

```text
data_cache/npy/
```

Script có resume: file `.npy` hoàn chỉnh sẽ được bỏ qua khi chạy lại. Không commit `data_cache/` lên Git.

Kiểm tra:

```bash
find data_cache/npy -name '*.npy' | wc -l
du -sh data_cache/npy
```

Số file mong đợi là 129 theo manifest. Thư mục `data/` hiện có thể có thêm một CSV không thuộc manifest; cache chỉ xử lý file được manifest tham chiếu.

## 5. Smoke test và audit

```bash
python paper/run_revision.py smoke
python paper/run_revision.py audit
```

Đọc audit:

```bash
sed -n '1,240p' paper/revision_artifacts/protocol_audit.md
```

## 6. Chạy thử một trial

```bash
python scripts/search_revision_5060.py search \
  --base-config configs/revision_search_5060.yaml \
  --output-root runs/revision_search_5060 \
  --trials 1 \
  --continue
```

Trong terminal khác, theo dõi:

```bash
nvidia-smi dmon -s pucm
```

Và:

```bash
tail -f runs/revision_search_5060/trial_001/console.log
```

Nếu batch 128 gây CUDA out-of-memory ở trial khác, trial đó được đánh dấu failed và search tiếp tục. Không cần giảm toàn bộ search space ngay.

## 7. Chạy queue liên tục ở background

Queue thực hiện theo thứ tự:

1. Hoàn tất/resume NPY cache.
2. Preflight, smoke và audit.
3. Search 24 trial multimodal theo validation.
4. Search 16 trial vibration-only theo validation.
5. Dừng trước test để người nghiên cứu kiểm tra leaderboard.

Cho phép script chạy khi màn hình khóa và chặn máy sleep:

```bash
mkdir -p runs/revision_queue

nohup setsid systemd-inhibit \
  --what=sleep:idle \
  --why="Bearing paper validation search" \
  bash paper/run_5060_search_queue.sh \
  > runs/revision_queue/search.log 2>&1 &

echo $! | tee runs/revision_queue/search.pid
```

Có thể đóng terminal sau đó.

Theo dõi log:

```bash
tail -f runs/revision_queue/search.log
```

Theo dõi GPU:

```bash
watch -n 2 nvidia-smi
```

Theo dõi leaderboard:

```bash
watch -n 15 tail -n 12 runs/revision_search_5060/leaderboard.csv
```

Kiểm tra queue còn chạy:

```bash
ps -fp "$(cat runs/revision_queue/search.pid)"
```

Nếu cần dừng có kiểm soát:

```bash
kill -INT -- "-$(cat runs/revision_queue/search.pid)"
```

Chạy lại đúng lệnh `nohup` ở trên để resume. Các trial có `result.json` sẽ được bỏ qua.

## 8. Chọn winner đúng cách

Sau khi queue kết thúc:

```bash
column -s, -t < runs/revision_search_5060/leaderboard.csv | head -12
column -s, -t < runs/revision_search_vibration_5060/leaderboard.csv | head -12
```

Winner đã được lưu tại:

```text
runs/revision_search_5060/best_search_config.yaml
runs/revision_search_vibration_5060/best_search_config.yaml
```

Không mở report test trong quá trình search vì search không sinh report test. Nếu muốn mở rộng search space, quyết định trước khi chạy confirmation.

## 9. Khóa config và xác nhận test qua 5 seed

Chỉ chạy bước này một lần sau khi search space đã chốt.

Multimodal:

```bash
python scripts/search_revision_5060.py confirm \
  --output-root runs/revision_search_5060 \
  --confirm-root runs/revision_confirm_5060 \
  --seeds 42,43,44,45,46 \
  --continue
```

Vibration-only:

```bash
python scripts/search_revision_5060.py confirm \
  --output-root runs/revision_search_vibration_5060 \
  --confirm-root runs/revision_confirm_vibration_5060 \
  --seeds 42,43,44,45,46 \
  --continue
```

SVM 8-D:

```bash
python classical_baselines/train_classical.py \
  --config classical_baselines/configs/revision_svm_vib8_stratified.yaml
```

## 10. Đọc kết quả dùng cho bài báo

```bash
cat runs/revision_confirm_5060/summary.md
cat runs/revision_confirm_vibration_5060/summary.md
sed -n '1,40p' runs/revision/svm_vib8_stratified/report_test.txt
```

Các bảng JSON đầy đủ:

```text
runs/revision_confirm_5060/aggregate.json
runs/revision_confirm_vibration_5060/aggregate.json
```

Paper nên báo cáo mean ± standard deviation của năm seed, không chỉ lấy seed có test score cao nhất.

## 11. Sửa bài báo

Thứ tự cập nhật `paper/main.tex`:

1. Primary result: multimodal multi-class held-out, mean ± std.
2. Baseline: vibration-only đã tune validation độc lập và SVM.
3. Ghi rõ split revision mới được cố định bằng seed `20260803`; năm seed 42–46 chỉ thay initialization/training randomness.
4. Ghi `samples_per_file`, batch size, AMP, TF32 và hardware RTX 5060 Ti.
5. Full-range cũ đổi tên thành whole-trajectory retrospective evaluation.
6. Temporal single-class slices chỉ để secondary analysis.
7. Không claim cross-bearing generalization vì dataset hiện chỉ có một run.

Sau khi sửa:

```bash
python paper/run_revision.py latex
```

## 12. Quy tắc để “số đẹp” vẫn hợp lệ khoa học

- Search chỉ nhìn validation Macro-F1.
- Dùng split seed revision mới `20260803` vì test seed 42 đã được xem trong các run lịch sử.
- Test chỉ chạy sau khi config đã khóa.
- Không chọn seed test tốt nhất; báo cáo mean ± std của toàn bộ seed định trước.
- Không đổi search space sau khi xem test. Nếu buộc phải đổi, cần tạo một test set mới chưa từng xem.
- Giữ lại mọi trial, config, console log và leaderboard.
- So sánh baseline trên cùng split, aggregation và test support.
