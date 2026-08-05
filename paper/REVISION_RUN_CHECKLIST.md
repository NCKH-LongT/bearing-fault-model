# Checklist chạy revision bài báo

Cập nhật gần nhất: **2026-08-05 23:17 ICT**.

File này là nhật ký vận hành và checklist resume cho quy trình revision. Kết quả chính phải đến từ held-out multi-class test ở cấp file. Hyperparameter chỉ được chọn bằng validation Macro-F1; không mở test trong lúc search.

## Trạng thái nhanh

- Queue v2: **đã hoàn thành**.
- Công việc hiện tại: classical validation robustness/efficiency đã hoàn thành; bước tiếp theo là deep-model robustness/latency/memory rồi sửa paper, hoặc bổ sung run/bearing mới.
- Search v1: hoàn thành 24/24 multimodal và 16/16 vibration-only, giữ lại làm audit trail nhưng không dùng cho confirmation vì có 23 cảnh báo scheduler/optimizer.
- Multimodal v2 đã hoàn thành: `24/24` trial và confirmation đủ 5 seed.
- Vibration-only v2 đã hoàn thành: `16/16` trial và confirmation đủ 5 seed.
- Cache: `129/129` file trong `data_cache/npy/`, khoảng 3.9 GB.
- GPU: NVIDIA GeForce RTX 5060 Ti; PyTorch `2.11.0+cu128`; CUDA khả dụng.
- Legacy checkpoint `runs/logs_stft_strat/auto_r22/best.pt`: **thiếu**. Không chạy secondary legacy reproduction.
- Test set: **đã mở đúng một lần sau khi khóa hai winner v2**; không tune thêm trên test hiện tại.

## Hướng dẫn chạy tuần tự

Chạy từ repository root và chỉ chuyển bước khi điều kiện cuối bước đạt. Các ô `[x]` đã hoàn thành trong lần chạy hiện tại; ô `[ ]` còn phải thực hiện.

### Bước 1 — Kiểm tra môi trường và dữ liệu

- [x] Tạo và dùng `.venv`.
- [x] Cài PyTorch CUDA và dependency của paper.
- [x] Xác nhận CUDA nhận RTX 5060 Ti.
- [x] Preflight: 129 manifest rows; 77 Healthy, 39 Degrading, 13 Fault.
- [x] LaTeX tools hoạt động.
- [x] Tạo NPY cache đủ 129 file.
- [x] Smoke test STFT `2x160x160`, temperature 6-D, forward và backward thành công.

Lệnh chạy/kiểm tra lại:

```bash
source .venv/bin/activate
python paper/run_revision.py preflight
python paper/run_revision.py smoke
find data_cache/npy -name '*.npy' | wc -l
```

**Điều kiện sang bước 2:** preflight báo `Training inputs OK` và `LaTeX tools OK`, smoke báo `OK`, cache có 129 file.

### Bước 2 — Audit protocol

- [x] Chạy `python paper/run_revision.py audit`.
- [x] Primary train/validation/test không giao file.
- [x] Primary test có đủ ba lớp: 16 Healthy, 8 Degrading, 3 Fault.
- [x] Ghi nhận legacy stratified train giao temporal test 25 file.
- [x] Ghi nhận full-range cũ chỉ là whole-trajectory retrospective evaluation.
- [x] Không dùng single-class temporal slices làm kết quả chính.

Audit: `paper/revision_artifacts/protocol_audit.md`.

Lệnh:

```bash
python paper/run_revision.py audit
sed -n '1,240p' paper/revision_artifacts/protocol_audit.md
```

**Điều kiện sang bước 3:** primary train/validation/test giao nhau 0 file và test có đủ ba lớp.

### Bước 3 — Validation-only hyperparameter search

- [x] Chạy smoke trial multimodal 001.
- [x] Search v1 hoàn thành nhưng được thay thế có kiểm soát sau khi phát hiện AMP có thể bỏ qua optimizer update trong khi scheduler vẫn bước.
- [x] Sửa `train_logs.py` để scheduler chỉ bước sau optimizer update thực sự.
- [x] Giữ nguyên artifact v1; tạo output root v2, không ghi đè.
- [x] Khởi động queue nền có resume và chống sleep.
- [x] Hoàn thành đủ 24 multimodal trials.
- [x] Kiểm tra `runs/revision_search_5060_v2/leaderboard.csv`.
- [x] Xác nhận winner tại `runs/revision_search_5060_v2/best_search_config.yaml`.
- [x] Hoàn thành đủ 16 vibration-only trials.
- [x] Kiểm tra `runs/revision_search_vibration_5060_v2/leaderboard.csv`.
- [x] Xác nhận winner tại `runs/revision_search_vibration_5060_v2/best_search_config.yaml`.
- [x] Khóa search space; không thay đổi sau khi xem test.

Queue tự dừng trước test. Không chạy thêm queue song song.

Theo dõi:

```bash
ps -fp "$(cat runs/revision_queue_v2/search.pid)"
tail -f runs/revision_queue_v2/search.log
watch -n 2 nvidia-smi
watch -n 15 tail -n 12 runs/revision_search_5060_v2/leaderboard.csv
```

Kiểm tra tiến độ mà không mở test:

```bash
python scripts/search_revision_5060.py status \
  --output-root runs/revision_search_5060_v2 --expected 24

python scripts/search_revision_5060.py status \
  --output-root runs/revision_search_vibration_5060_v2 --expected 16
```

Queue hoàn thành khi cuối `runs/revision_queue_v2/search.log` có dòng:

```text
Validation searches finished.
```

**Điều kiện sang bước 4:** status của cả hai search báo `Search complete: yes` và `Winner locked: yes`. Không dùng file `best_search_config.yaml` tạm thời khi search chưa đủ trial.

Nếu queue bị gián đoạn, resume bằng:

```bash
REVISION_MULTIMODAL_ROOT=runs/revision_search_5060_v2 \
REVISION_VIBRATION_ROOT=runs/revision_search_vibration_5060_v2 \
nohup setsid systemd-inhibit \
  --what=sleep:idle \
  --why="Bearing paper validation search" \
  bash paper/run_5060_search_queue.sh \
  > runs/revision_queue_v2/search.log 2>&1 &

echo $! | tee runs/revision_queue_v2/search.pid
```

### Bước 4 — Locked test confirmation

Chỉ thực hiện sau khi mục C hoàn thành và search space đã khóa.

- [x] Confirm multimodal qua seeds 42–46.
- [x] Confirm vibration-only qua seeds 42–46.
- [x] Chạy SVM 8-D trên cùng split.
- [x] Báo cáo mean ± standard deviation, không chọn seed test tốt nhất.
- [x] Kiểm tra test support: 27 file gồm 16 Healthy, 8 Degrading, 3 Fault.

Multimodal:

```bash
source .venv/bin/activate
python scripts/search_revision_5060.py confirm \
  --output-root runs/revision_search_5060_v2 \
  --confirm-root runs/revision_confirm_5060_v2 \
  --seeds 42,43,44,45,46 \
  --continue
```

Vibration-only:

```bash
python scripts/search_revision_5060.py confirm \
  --output-root runs/revision_search_vibration_5060_v2 \
  --confirm-root runs/revision_confirm_vibration_5060_v2 \
  --seeds 42,43,44,45,46 \
  --continue
```

SVM:

```bash
python classical_baselines/train_classical.py \
  --config classical_baselines/configs/revision_svm_vib8_stratified.yaml
```

Kết quả cần đọc:

```bash
cat runs/revision_confirm_5060_v2/summary.md
cat runs/revision_confirm_vibration_5060_v2/summary.md
sed -n '1,40p' runs/revision/svm_vib8_stratified/report_test.txt
```

**Điều kiện sang bước 5:** hai `summary.md` có đủ năm seed, SVM có `report_test.txt`, và test support giống nhau.

### Bước 5 — Tổng hợp evidence tối thiểu cho paper

- [x] Lập bảng Accuracy, Macro-F1 và F1 từng lớp cho SVM, vibration-only và multimodal.
- [x] Báo cáo mean ± std của năm seed deep model.
- [x] Tính bootstrap 95% CI ở cấp file với 10.000 replicate.
- [x] Thực hiện paired bootstrap và McNemar exact ở cấp file.
- [x] Chẩn đoán prediction collapse, seed agreement, validation–test gap và thang đo temperature feature.
- [x] Bổ sung temperature-only baseline bằng scaler fit trên train; locked test không được đánh giá.
- [x] Chạy five-seed multimodal với temperature normalization fit trên train.
- [x] Audit balanced sampler và gradient theo lớp chỉ bằng train split.
- [x] Xác nhận manifest chỉ có một `run1`; không giả lập leave-one-run-out/cross-bearing.
- [x] Chọn SVM C/gamma bằng five-fold stratified file-grouped CV trên train, mean decision-score aggregation.
- [x] So sánh vibration 8-D/26-D với SVM, Logistic Regression và Random Forest trên cùng file folds.
- [x] So sánh handcrafted vibration 26-D với vibration+temperature 32-D trên cùng folds.
- [x] Chạy temperature-only 6-D file-CV và file-grouped modality permutation/TTF-correlation audit.
- [x] Đo validation robustness và CPU efficiency cho classical P2 winner.
- [ ] Bổ sung efficiency: params, model size, STFT/model/end-to-end latency, throughput và memory.
- [ ] Bổ sung robustness: vibration noise, temperature missing/drift và mất một vibration axis.
- [ ] Nếu không có thêm run/bearing, hạ claim về within-run file-level classification.

Kết quả held-out test hiện tại:

| Model | Accuracy | Macro-F1 | Healthy F1 | Degrading F1 | Fault F1 |
|---|---:|---:|---:|---:|---:|
| SVM vibration 8-D | 0.8148 | 0.7681 | 0.8889 | 0.6154 | 0.8000 |
| Vibration-only CNN | 0.4593 ± 0.2437 | 0.3314 ± 0.2643 | 0.4118 ± 0.3858 | 0.3425 ± 0.3204 | 0.2400 ± 0.4336 |
| Multimodal late fusion | 0.5556 ± 0.1889 | 0.4544 ± 0.1138 | 0.4932 ± 0.4512 | 0.3500 ± 0.2049 | 0.5200 ± 0.3271 |

Diễn giải và quyết định:

- Báo cáo `paper/revision_artifacts/deep_collapse_diagnostics/diagnostics.md` xác nhận 9/10 lượt deep bỏ hẳn ít nhất một lớp, 0/27 file có dự đoán nhất trí giữa năm seed và validation–test Macro-F1 gap `0.2797–0.9333`.
- Temperature descriptor cũ chưa được chuẩn hóa theo train và có raw train standard-deviation scale ratio `2768.14`. Pipeline mới đã thêm scaler fit trên train và lưu mean/std trong run-local config.
- Temperature-only MLP `6→32→3` (323 tham số) trên five-seed validation đạt Accuracy `0.7385 ± 0.0322`, Macro-F1 `0.7776 ± 0.0264`; cả năm seed dự đoán đủ ba lớp. Artifact: `paper/revision_artifacts/temperature_only_validation/`.
- Multimodal normalized đạt Accuracy/Macro-F1/F1 từng lớp `1.0000 ± 0.0000` trên five-seed validation. Sampler draw gần đều `34.11%/32.54%/33.35%`; mọi nhánh có gradient RMS khác 0 cho cả ba lớp.
- Đây là kết quả exploratory validation, không được đặt cạnh locked-test SVM/deep như một so sánh test trực tiếp. Validation đã bão hòa nên không thể dùng để chứng minh normalization cải thiện generalization; cần group-CV/validation run mới và tuyệt đối không chọn cấu hình bằng locked test đã xem.
- SVM file-grouped CV winner là `C=1`, `gamma=0.1`, mean decision score. Train-CV Macro-F1 `0.8157 ± 0.1576`; validation Accuracy `0.8462`, Macro-F1 `0.8631`. Đây vẫn là within-run CV vì manifest chỉ có `run1`; fold thấp nhất `0.5582` cho thấy chưa ổn định theo file-group.
- Feature/algorithm winner là SVM vibration 26-D `C=0.1`, `gamma=0.1`, CV Macro-F1 `0.8416 ± 0.0914`; bộ 8-D tốt nhất đạt `0.8157 ± 0.1576`. Logistic Regression 26-D tốt nhất đạt `0.8295 ± 0.1597`, Random Forest tốt nhất `0.7904 ± 0.1740`. Global CV winner được đánh giá validation đúng một lần và đạt 1.0000; không xem đây là test evidence.
- Khi thêm temperature stats 6-D, Random Forest 32-D trở thành global winner: CV Macro-F1 `0.9634 ± 0.0337`, tăng `0.1218` so với vibration 26-D winner và fold thấp nhất `0.9348`. Validation đạt 1.0000 nhưng không phải test evidence. Vì chỉ có `run1`, cần kiểm tra temperature có đang proxy cho TTF hay không.
- Temperature-only Random Forest đạt CV Macro-F1 `0.8954 ± 0.0694`. Permutation vibration/temperature làm fusion Macro-F1 giảm lần lượt `0.3101/0.3489`, nên cả hai modality đều đóng góp trong run1. Bearing mean/std/slope có Spearman rho với TTF `0.8714/0.7988/0.7390`, xác nhận nguy cơ trajectory proxy; không claim cross-run.
- Classical robustness: noise 20/10 dB giữ Macro-F1 1.0000 trên validation bão hòa; mất vibration X, mất temperature và drift +2°C làm giảm lần lượt `0.4946/0.4470/0.4367`; mất vibration Y không đổi. Feature/RF latency CPU `2.458/0.0363 ms/window`, khoảng `79.82 ms/file` cho 32 window, model `2253.4 KiB`. Deep robustness/latency/memory vẫn chưa hoàn thành.
- SVM hiện là model mạnh nhất trên test này; Macro-F1 cao hơn multimodal trung bình 0.3137 và vibration-only trung bình 0.4367.
- Multimodal tốt hơn vibration-only trung bình nhưng cả hai deep model rất không ổn định qua seed và có hiện tượng bỏ hẳn một lớp.
- SVM dùng cấu hình cố định `C=1`, RBF, `gamma=scale`, balanced class weight, StandardScaler và mean-probability file aggregation. Cấu hình này chưa được tune trên validation; vì test đã được xem, không được tune C/gamma hoặc chọn thuật toán mới dựa trên kết quả test hiện tại rồi tiếp tục báo cáo như confirmation độc lập.
- `SVC(probability=True)` hiệu chỉnh xác suất bằng CV nội bộ ở cấp window; các window tương quan từ cùng file có thể rơi vào các fold calibration khác nhau. Đây không phải test leakage, nhưng mean-probability aggregation có thể lạc quan. Lần protocol mới nên so sánh mean decision score hoặc dùng calibration group-aware theo file.
- Nếu bổ sung Logistic Regression, Random Forest, XGBoost hoặc validation grid search, phải ghi là post-hoc exploratory; để dùng làm kết quả confirmatory cần khóa trước trên validation và đánh giá trên một test set/run mới chưa từng xem.
- Ưu tiên tiếp theo không phải tối ưu SVM để lấy số cao hơn, mà là bootstrap CI/file-level comparison, phân tích lỗi deep model, và bổ sung run/bearing độc lập nếu có.

**Điều kiện sang bước 6:** bảng kết quả chỉ dùng artifact mới, có mean ± std và mọi model được so sánh trên cùng split/test support.

### Bước 6 — Sửa và biên dịch bài báo

- [ ] Abstract dùng primary held-out multi-class result.
- [ ] Contributions mô tả compact late fusion, không overclaim kiến trúc mới.
- [ ] Experimental Setup ghi split seed `20260803`, training seeds 42–46 và class support.
- [ ] Results theo thứ tự primary → baselines → robustness/efficiency → retrospective trajectory.
- [ ] Discussion và Limitations ghi rõ single-run và chưa có cross-bearing generalization.
- [ ] Full-range cũ được gọi là whole-trajectory retrospective evaluation.
- [ ] Chỉ cập nhật số trong `paper/main.tex` từ artifact mới đã xác minh.
- [ ] Biên dịch `paper/main.pdf` thành công.

Lệnh biên dịch:

```bash
python paper/run_revision.py latex
```

**Điều kiện hoàn thành:** `paper/main.pdf` biên dịch thành công và mọi con số trong LaTeX truy ngược được tới report/JSON đã lưu.

## Ghi chú vận hành

- Search v1 có 23 cảnh báo `lr_scheduler.step()` trước optimizer update do AMP overflow. Lỗi đã sửa sau khi v1 hoàn tất và trước khi mở test. Search v2 chạy lại toàn bộ search space; chỉ artifact v2 được phép đi vào confirmation/paper.
- Trial validation đạt 1.0 không có nghĩa test đạt 1.0. Validation chỉ có 26 file và Fault support là 3, nên cần five-seed confirmation và confidence interval.
- Không chạy `paper/run_revision.py secondary` khi thiếu checkpoint `auto_r22`; ngay cả khi có checkpoint, kết quả đó chỉ là secondary retrospective analysis.

## Nhật ký

- **2026-08-04 19:39 ICT:** trial multimodal 001 hoàn thành; validation Macro-F1 1.0000.
- **2026-08-04 19:43 ICT:** khởi động validation queue nền, PID 20903.
- **2026-08-04 19:47 ICT:** đọc lại toàn bộ Markdown trong `paper`; xác nhận queue đang chạy trial 002, cache/preflight/smoke/audit đều đạt; test vẫn khóa.
- **2026-08-04 20:10 ICT:** thực hiện kiểm tra leaderboard lần 1; multimodal hoàn thành 5/24 trial và đang chạy trial 006. Bốn trial đang đồng hạng validation Macro-F1 1.0000; vibration-only chưa bắt đầu; chưa đủ điều kiện khóa winner hoặc mở test.
- **2026-08-04 20:14 ICT:** chạy lại bước kiểm tra; multimodal 5/24, vibration-only 0/16, chưa khóa winner. Bổ sung lệnh `search_revision_5060.py status` và viết lại checklist thành sáu bước có điều kiện chuyển bước.
- **2026-08-04 20:15 ICT:** chạy lại Bước 1 theo hướng dẫn mới; preflight đạt, LaTeX tools đạt, smoke forward/backward đạt và cache đủ 129/129. Validation queue vẫn chạy độc lập.
- **2026-08-05 00:43 ICT:** search v1 hoàn thành 24/24 multimodal và 16/16 vibration-only; cả hai leader là trial 001 với validation Macro-F1 1.0000; test vẫn khóa.
- **2026-08-05 00:44 ICT:** xác nhận 23 cảnh báo scheduler trong v1; sửa AMP step detection, kiểm tra cú pháp/smoke thành công và khởi động search v2 tại `runs/revision_search_5060_v2` cùng `runs/revision_search_vibration_5060_v2`, PID 314569.
- **2026-08-05 11:32 ICT:** xác minh confirmation v2 đủ năm seed cho multimodal và vibration-only; xác minh SVM test support 16/8/3 và kết quả Accuracy 0.8148, Macro-F1 0.7681. SVM vượt hai deep model; ghi rõ không tune thêm trên test đã mở và chuyển trọng tâm sang CI, error analysis và external/run-disjoint validation.
- **2026-08-05 11:52 ICT:** hợp nhất hướng dẫn hiện hành vào `paper/README.md`, thêm roadmap P0–P5, thêm script đóng gói winner artifact và chuẩn bị GitHub Release `revision-v2-artifacts-20260805`. Xóa các guide trong `paper/` đã bị tài liệu canonical thay thế.
- **2026-08-05 19:48 ICT:** xuất file ID, mean logits/probabilities và prediction cho 10 checkpoint deep; xuất file ID/score cho SVM; chạy 10.000 file-level bootstrap replicate, paired bootstrap và McNemar exact. Báo cáo lưu tại `paper/revision_artifacts/locked_test_statistics/`.
- **2026-08-05 20:02 ICT:** chạy chẩn đoán deep collapse. Xác nhận 9/10 seed bỏ ít nhất một lớp, seed agreement rất thấp, validation–test gap lớn và temperature feature chưa được scale. Lưu báo cáo tái lập tại `paper/revision_artifacts/deep_collapse_diagnostics/`; khóa test khỏi mọi quyết định nâng cấp tiếp theo.
- **2026-08-05 20:15 ICT:** triển khai train-only temperature standardization và temperature-only MLP không tính STFT; chạy seeds 42–46 chỉ trên validation. Kết quả Accuracy `0.7385 ± 0.0322`, Macro-F1 `0.7776 ± 0.0264`, không seed nào bỏ lớp. Lưu artifact tại `paper/revision_artifacts/temperature_only_validation/`; locked test không được gọi.
- **2026-08-05 20:43 ICT:** chạy multimodal train-normalized seeds 42–46 chỉ trên validation; cả năm seed đạt Macro-F1 1.0000. Audit train-only xác nhận balanced sampler gần đều và cả ba nhánh nhận gradient từ mọi lớp. Validation đã bão hòa, vì vậy quyết định chuyển sang group-CV/run độc lập thay vì mở lại locked test.
- **2026-08-05 22:44 ICT:** xác nhận dataset chỉ có một `run1`, không đủ leave-one-run-out. Chạy grid 16 SVM bằng five-fold stratified file-grouped CV trên train và mean decision aggregation; winner `C=1`, `gamma=0.1`, CV Macro-F1 `0.8157 ± 0.1576`, validation Macro-F1 `0.8631`. Locked test không được khởi tạo.
- **2026-08-05 22:49 ICT:** thêm vibration feature 26-D và so sánh 48 tổ hợp feature/model trên cùng five file-folds. Winner SVM 26-D `C=0.1`, `gamma=0.1` đạt CV Macro-F1 `0.8416 ± 0.0914`; global winner đạt validation 1.0000. Locked test không được khởi tạo; bước tiếp theo là vibration+temperature handcrafted ablation.
- **2026-08-05 22:57 ICT:** mở rộng ablation lên 72 cấu hình bằng handcrafted vibration+temperature 32-D. Random Forest depth 12/leaf 1 đạt CV Macro-F1 `0.9634 ± 0.0337`, vượt vibration-only winner `0.8416 ± 0.0914`; validation 1.0000. Locked test không được khởi tạo; ghi rõ nguy cơ temperature proxy cho TTF trong single-run trajectory.
- **2026-08-05 23:10 ICT:** mở rộng lên 96 cấu hình với temperature-only 6-D; temperature-only RF đạt CV Macro-F1 `0.8954 ± 0.0694`. Chạy 30 file-grouped permutation: tráo vibration/temperature làm Macro-F1 giảm `0.3101/0.3489`; bearing temperature mean tương quan TTF `ρ=0.8714`. P2 được khóa; không tiếp tục model shopping trên run1.
- **2026-08-05 23:17 ICT:** chạy validation-only robustness/efficiency cho Random Forest 32-D. Noise 20/10 dB không giảm điểm; mất vibration X, temperature, hoặc drift +2°C gây giảm Macro-F1 lớn. Feature extraction `2.458 ms/window`, inference `0.0363 ms/window`, model khoảng 2.2 MiB. Locked test không được khởi tạo.
