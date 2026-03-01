# Ghi chú chỉnh sửa bài (Status & TODO)

## Tình trạng hiện tại
- Figure placeholder: còn 1 hình minh họa dạng placeholder ở mục Ablation (`fig:abl_slice`).
- Related Works: đã được lược bỏ và gộp nội dung cốt lõi vào phần Introduction (không còn mục riêng).

## Tuỳ chọn xử lý
- Placeholder (`fig:abl_slice`)
  - Thay bằng hình thật: chèn 2 ảnh confusion (broad slice `[70,100.1]%` và very‑late `[90,100.1]%`), giữ nguyên `\label{fig:abl_slice}`.
  - Hoặc xoá figure: gỡ block hình placeholder và sửa câu chữ để không còn tham chiếu tới hình.
- Related Works
  - Giữ như hiện tại: nội dung chính (leakage, đa cảm biến) đã nằm trong Introduction.
  - Hoặc thêm mục “Related Work” ngắn (4–6 câu) nhấn mạnh: CNN time–frequency cho chẩn đoán, nguy cơ leakage và TTF‑split, lợi ích đa cảm biến; trích dẫn ngắn gọn (ví dụ: zhao2019deep, lei2020roadmap, ince2016tie, wen2018neurocomputing, kaufman2012leakage, bergmeir2012timeseriescv, baltrusaitis2019multimodal).

## Đề xuất hành động
1) Chọn 1 trong 2 phương án cho placeholder:
   - A. Thay bằng hình thật (cần cung cấp ảnh very‑late `[90,100.1]%` nếu chưa có sẵn).
   - B. Xoá figure và tinh giản câu chữ liên quan.
2) (Tuỳ chọn) Thêm mục “Related Work” ngắn để tăng tính hoàn chỉnh biên tập.

## Ghi chú xác minh nhanh
- Đường dẫn hình ảnh đang sử dụng đều tồn tại: `figures/stratified/*`, `figures/temporal/*`, `figures/temporal_alltest/*`, `figures/temporal_broad/*`.
- Các tham chiếu nhãn chính hợp lệ: `fig:strat_pair`, `fig:temporal_pair`, `fig:protocol_compare`, `fig:abl_slice`, `tab:settings`, `tab:abl_summary`.
- Hai hình có nhãn nhưng chưa được gọi trong câu chữ (không bắt buộc): `fig:temporal_alltest_vote`, `fig:temporal_broad_pair`.

## Nếu cần mình thực hiện ngay
- A) Thay placeholder bằng hình thật (khi có file ảnh cung cấp) hoặc
- B) Gỡ placeholder và cập nhật `main.tex` cho gọn.
- C) Thêm mục “Related Work” ngắn (4–6 câu) vào sau Introduction.
