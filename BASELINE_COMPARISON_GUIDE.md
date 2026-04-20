# Baseline Comparison Guide

Tài liệu này ghi lại hướng đi so sánh khi tạo một model mới cho bài toán phân loại 3 giai đoạn ổ bi trong repo này.

## Mục tiêu

Khi có model mới, câu hỏi không phải chỉ là "model này có điểm cao không", mà là:

- Nó có vượt baseline cổ điển không?
- Nó có vượt baseline deep chỉ dùng vibration không?
- Nó có vượt model tốt nhất hiện tại của repo không?
- Nó có còn tốt khi dùng đúng protocol chống leakage không?

Nếu không trả lời được 4 câu hỏi đó, phần so sánh sẽ yếu.

## Thứ tự baseline nên so sánh

### 1. Baseline bắt buộc

Đây là bộ tối thiểu nên có trong hầu hết các lần thử model mới:

- `Classical vib (SVM, 8D)`
- `Vibration-only CNN`
- `Current best multi-modal (vib + temp, late fusion)`
- `Your new model`

Lý do:

- `SVM 8D` cho thấy model mới có thật sự hơn ML cổ điển hay không.
- `Vibration-only CNN` cho thấy phần cải tiến của bạn có thực sự hơn deep baseline đơn cảm biến hay không.
- `Current best multi-modal` là baseline quan trọng nhất, vì đó là model mạnh nhất hiện có trong bài.

### 2. Baseline gần kiến trúc

Nếu model mới vẫn cùng họ với pipeline hiện tại, nên so sánh thêm:

- Cùng backbone nhưng bỏ temperature branch
- Cùng backbone nhưng bỏ transfer learning / stratified initialization
- Cùng backbone nhưng bỏ temporal fine-tuning
- Cùng backbone nhưng thay aggregation (`mean-logit` vs `vote`) nếu aggregation là một phần đóng góp

Mục tiêu của nhóm này là tách phần "mới" ra khỏi phần "đã có sẵn".

### 3. Baseline từ Related Work

Khi viết bài hoàn chỉnh hoặc nộp hội nghị/tạp chí, có thể bổ sung:

- `STFT/CWT + CNN`
- `Raw-signal to image + CNN`
- `Transfer learning baseline`
- `Multimodal fusion baseline`

Nhưng các baseline này chỉ có giá trị nếu chạy lại được trên cùng dữ liệu và cùng protocol.

## So sánh nào là đủ?

### Trường hợp 1: chỉ muốn chứng minh model mới tốt hơn repo hiện tại

Chỉ cần:

- `SVM 8D`
- `Vibration-only CNN`
- `Current best multi-modal`
- `New model`

Đây là bộ thực tế nhất.

### Trường hợp 2: muốn viết bài mạnh hơn

Nên có:

- Các baseline bắt buộc ở trên
- 1 đến 2 baseline từ literature nếu tái lập được
- Ablation để chỉ ra thành phần mới đóng góp ở đâu

## Có cần chạy model cổ điển trên cùng dataset với model mới không?

Có, nếu bạn muốn so sánh công bằng và đưa vào bài báo.

Không nên lấy kết quả `SVM` từ nơi khác rồi đặt cạnh model mới nếu:

- khác dataset
- khác manifest
- khác cách chia train/val/test
- khác windowing
- khác cách tính metric

Khi đó kết quả không còn so sánh trực tiếp được.

## Khi nào không cần chạy lại model cổ điển?

Không nhất thiết phải chạy lại ngay nếu bạn đang ở giai đoạn nghiên cứu nội bộ rất sớm, ví dụ:

- chỉ đang kiểm tra model mới có train được không
- chỉ đang xem loss/overfit
- chỉ đang test ý tưởng kiến trúc

Trong giai đoạn này, chỉ cần so với `current best multi-modal` hoặc `vibration-only CNN` là đủ để lọc ý tưởng.

Nhưng trước khi chốt kết quả để viết bài, nên chạy lại baseline cổ điển trên cùng protocol.

## Quy tắc công bằng khi so sánh

Mọi model đem ra so sánh nên dùng cùng:

- `data/manifest.csv`
- cùng labels
- cùng windowing
- cùng split logic
- cùng slice TTF
- cùng đơn vị đánh giá file-wise
- cùng metric

Cụ thể trong repo này, cần cố định:

- Stratified split: file-wise `0.6 / 0.2 / 0.2`
- Temporal split: train `[0,60]%`, val `[60,70]%`, test `[70,100]%`
- File-wise aggregation cho deep model
- Present-class scoring cho các slice bị thiếu lớp

## Deep model và classical model có nhất thiết phải dùng đúng cùng feature không?

Không.

Điều bắt buộc là dùng cùng dữ liệu và cùng protocol đánh giá. Feature có thể khác:

- Deep model có thể dùng STFT image + temp features
- Classical model có thể dùng handcrafted vibration features

Điều này vẫn hợp lệ, vì bạn đang so sánh các phương pháp khác nhau trên cùng bài toán.

Tuy nhiên cần mô tả rõ:

- classical dùng feature gì
- deep dùng feature gì
- cả hai được train/test trên cùng manifest và cùng split

## Cách chạy model cổ điển cho cùng dataset

Hướng đúng là không tách riêng dataset cho SVM. Thay vào đó:

1. Dùng cùng `manifest.csv`
2. Dùng cùng quy tắc chia file theo stratified hoặc temporal
3. Trích feature classical từ chính các file đó
4. Train SVM trên train split
5. Đánh giá trên đúng val/test split tương ứng
6. Báo cáo metric theo cùng logic của bài

Nói ngắn gọn:

- chung `samples`
- chung `splits`
- khác `features/model family`

## Repo này đã có gì sẵn

Các thành phần đã thấy trong repo:

- Temporal split / stratified split: `datasets/logs_ttf.py`
- Deep evaluation theo file: `eval_logs.py`
- Vibration-only ablation configs:
  - `configs/ablation/logs_stft_train_strat_vib.yaml`
  - `configs/ablation/logs_stft_temporal_vib_ft.yaml`
  - `configs/ablation/logs_stft_temporal_vib_ft_85100.yaml`
- Tổng hợp ablation/report:
  - `scripts/aggregate_ablation.py`
  - `scripts/auto_train_eval.py`

Điều còn thiếu rõ ràng là một pipeline chuẩn hóa cho `classical SVM` chạy chung split với deep models.

## Khuyến nghị triển khai classical baseline trong repo

Nên làm một pipeline riêng nhưng dùng lại split logic hiện có:

- `scripts/train_classical_baseline.py`
- `scripts/eval_classical_baseline.py`

Pipeline đó nên:

- đọc cùng `manifest`
- gọi lại logic split từ `datasets/logs_ttf.py` hoặc một hàm tách riêng
- sinh feature classical theo file/window
- train `SVM`
- xuất prediction theo file
- tính metric giống `eval_logs.py`

## Kết luận ngắn

Nếu model mới là ứng viên nghiêm túc, bạn nên ít nhất so sánh với:

- `SVM 8D`
- `Vibration-only CNN`
- `Current best multi-modal`

Và có, baseline cổ điển nên chạy lại trên cùng dataset và cùng protocol nếu bạn muốn kết quả có giá trị khoa học.

Nếu chỉ đang thử ý tưởng ban đầu, chưa cần chạy classical ngay; nhưng trước khi chốt kết quả cho bài báo thì nên có.
