# Artifact Của Paper

Thư mục này chứa các artifact đã được chọn lọc để đồng bộ paper với codebase hiện tại.

## Các thư mục chuẩn

- `stratified/`
  - Artifact tham chiếu cho pha phát triển, lấy từ `runs/logs_stft_strat/auto_r22/eval`
- `temporal/`
  - Artifact temporal theo hướng triển khai, được sync từ `runs/paper_sync/temporal/eval`
- `fullrange/`
  - Artifact full-range của paper, được sync từ `runs/paper_sync/fullrange/eval_vote`
- `ablation/`
  - Artifact hỗ trợ cho ablation, vẫn có thể được trích trong các bảng discussion

## Các con số chính hiện tại

- `temporal/report.txt`
  - Accuracy `0.8974`
- `temporal/report_early_70_90.txt`
  - Accuracy `0.8846`
  - F1 của `degrading` là `0.9388`
- `temporal/report_late_90_100.txt`
  - Accuracy `0.9231`
  - F1 của `fault` là `0.9600`
- `fullrange/report.txt`
  - Accuracy `0.8915`
  - Macro-F1 `0.8739`

## Các thư mục lưu trữ

Một số artifact top-level cũ chỉ được giữ lại để tham khảo đã được chuyển vào `archive/` nhằm giảm nhiễu:

- `archive/temporal_alltest/`
- `archive/temporal_broad/`
- `archive/ablation_vote_report.txt`

## Tái sinh lại

Dùng lệnh:

```powershell
python scripts/run_paper_sync.py --python .venv/Scripts/python.exe --sync-figures
```

Để xem toàn bộ quy trình, đọc `docs/PAPER_RERUN_GUIDE.md`.
