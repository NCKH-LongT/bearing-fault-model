# Danh Mục Config

Thư mục này chỉ giữ các config cần thiết cho workflow reviewer hiện tại.

## Các config chuẩn để dùng cho paper

Nếu cần một bộ chuẩn duy nhất để viết paper và chạy lại kết quả, hãy dùng các file sau:

- `best_stratified_ref.yaml`
- `best_temporal.yaml`
- `best_fullrange_eval.yaml`

Ba file này là lớp chuẩn hóa cuối cùng dựa trên bộ run tốt nhất hiện đang được dùng cho paper.

## Các config paper-sync nền

Các file dưới đây vẫn hợp lệ và là nền để sinh bộ config chuẩn phía trên:

- `paper_sync_stratified.yaml`
- `paper_sync_temporal.yaml`
- `paper_sync_fullrange.yaml`

## Các config gắn với thí nghiệm cụ thể

Phần lớn các file này thuộc về các nhánh ablation hoặc search cũ, không nên dùng làm pipeline mặc định cho paper:

- `ablation/*.yaml`

## Quy tắc sử dụng nhanh

- Nếu mục tiêu là "chạy lại paper ngay bây giờ", hãy bắt đầu từ `best_*.yaml`.
- Nếu mục tiêu là "xem lại pipeline paper-sync gốc", dùng `paper_sync_*.yaml`.
- Nếu mục tiêu là "xem lại ablation hoặc đối chiếu baseline", hãy dùng các config trong `configs/ablation/` một cách có chủ đích và ghi rõ lựa chọn đó.
