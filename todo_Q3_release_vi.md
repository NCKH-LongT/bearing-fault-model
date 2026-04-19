Kế Hoạch Hành Động Chuẩn Bị Nộp Q3 (TODO)

Mục tiêu: Nâng cấp bản thảo và hiện vật kèm theo lên mức sẵn sàng nộp Q3 với bằng chứng khái quát hóa mạnh hơn, thống kê nghiêm ngặt, định nghĩa chỉ số rõ ràng và khả năng tái lập đầy đủ.

Ưu Tiên Cao (trước khi nộp)
- Khả năng khái quát ra ngoài tập huấn luyện
  - [ ] Bổ sung đánh giá cross-run/cross-bearing trên cùng bộ dữ liệu (nếu có).
  - [ ] Thêm ít nhất một bộ dữ liệu ngoài (vd: Paderborn/CWRU/XJTU-SY) để kiểm tra chuyển giao (train trên Jung et al., test trên bộ ngoài; hoặc fine-tune few-shot).
  - [ ] Báo cáo ≥5 seed với trung bình ± độ lệch chuẩn cho các chỉ số chính; kèm khoảng tin cậy 95%.
  - [ ] Thêm kiểm định ý nghĩa cho cặp mô hình (ví dụ: McNemar) trên dự đoán ở mức tệp so với các baseline chính.

- Baseline mạnh và các biến thể
  - [ ] Triển khai baseline mạnh: 1D CNN và/hoặc TCN/LSTM trên rung thô; ResNet18 trên STFT; transformer thời gian đơn giản; tiền huấn luyện tự giám sát (SimCLR/CPC) + linear probe.
  - [ ] So sánh các chiến lược trộn (fusion): early, mid, late và gating/attention; giữ số tham số tương đồng khi có thể.
  - [ ] Đảm bảo có baseline vib-only mạnh song song mô hình đa cảm biến.

- Chỉ số “present-class” — định thức hóa và minh bạch
  - [ ] Thêm định nghĩa chính thức cho chỉ số “present-class” và cách xử lý lớp vắng mặt.
  - [ ] Cung cấp pseudo-code cho gộp mean-logit theo file, precision/recall/F1 cho lớp hiện diện.
  - [ ] Kèm ví dụ minh họa nhỏ ở phụ lục để tránh mơ hồ.

- Ngưỡng phân giai đoạn và phân tích ranh giới
  - [ ] Phân tích độ nhạy với ngưỡng TTF% quanh ranh Healthy–Degrading (vd: quét 0.55–0.70).
  - [ ] Cân nhắc bài toán thứ bậc (ordinal) hoặc loss hồi quy TTF phụ trợ để tăng vững; thảo luận ưu/nhược.
  - [ ] Gắn ngưỡng với quyết định vận hành (cảnh báo giai đoạn rất muộn) và biện minh.

- Hiệu năng và tài nguyên
  - [ ] Báo cáo số tham số, FLOPs (vd: ptflops/torchprofile) và độ trễ suy luận (ms/window) trên RTX 2060.
  - [ ] Tạo bảng gọn so sánh hiệu năng vs. độ chính xác giữa các baseline.

- Tái lập và hiện vật
  - [ ] Thay “Code will be released upon acceptance” bằng repo ẩn danh hoặc gói bổ sung (supplement) phục vụ review.
  - [ ] Bao gồm cấu hình chính xác, script tiền xử lý và một checkpoint nhỏ có thể tải.
  - [ ] Cố định phiên bản gói (requirements.txt/conda env); nêu rõ CUDA/PyTorch.
  - [ ] Cung cấp script chạy một lệnh cho dev split và temporal split (train/eval) kèm seed.
  - [ ] Ghi chú cờ tính xác định (cudnn.benchmark = False, cudnn.deterministic = True nếu áp dụng) và lưu ý.

- Hình và bảng
  - [ ] Hoàn thiện bảng ablation (điền ô trống hoặc đánh dấu N/A rõ ràng); làm rõ khi chỉ số là single-class.
  - [ ] Thêm PR/ROC cho lớp Fault (các lát temporal) và biểu đồ hiệu chỉnh (reliability/ECE).
  - [ ] Thêm spectrogram đại diện (Healthy/Degrading/Fault) và xu hướng nhiệt độ minh họa.
  - [ ] Đảm bảo mọi hình được tham chiếu đều tồn tại và đúng đường dẫn LaTeX.

- Viết, định dạng và tài liệu tham khảo
  - [ ] Mở rộng Related Work (fusion cho PHM; đánh giá tránh rò rỉ thời gian) với 6–10 trích dẫn trọng tâm.
  - [ ] Đổi tiêu đề “Related Works” → “Related Work”; bỏ chú thích tiếng Việt trong khối TikZ.
  - [ ] Làm rõ hạn chế (single-run) và cách thí nghiệm mới khắc phục.
  - [ ] Rà soát refs.bib về tính nhất quán (year/volume/pages/doi), bỏ urldate tương lai; giữ nguyên dạng viết hoa cần thiết.

Ưu Tiên Trung Bình
- Phân tích lỗi và khả năng giải thích
  - [ ] Phân tích lỗi gần 60–70% TTF; vẽ TTF vs. xác suất lớp; confusion theo thời gian.
  - [ ] Thêm saliency/Grad-CAM trên spectrogram hoặc SHAP cho đặc trưng nhiệt độ.

- Bổ sung ablation
  - [ ] Tóm lược hiệu ứng: bật/tắt class-weights, độ phân giải spectrogram, bật/tắt chuẩn hóa theo tần số, thay đổi độ dài/chéo cửa sổ.
  - [ ] Phần sweep siêu tham số ngắn với 3 yếu tố nhạy nhất.

- Kiểm thử độ vững
  - [ ] Nhiễu/điều kiện vận hành thay đổi (SNR, tốc độ/tải nếu có) để chứng minh lợi ích đa cảm biến.

Ưu Tiên Thấp / Nên Có
- [ ] Notebook demo tối giản (suy luận mức file, vẽ đồ thị) và ảnh chụp UI nhỏ.
- [ ] Ngưỡng bất định cho triển khai (reject option) và ablation.
- [ ] Ablation về chiều descriptor nhiệt độ và làm mượt.

Sửa Trực Tiếp Ở Mức File (bản thảo)
- main.tex
  - [ ] Sửa tiêu đề và mở rộng Related Work: main.tex:56.
  - [ ] Định nghĩa chỉ số present-class với công thức và pseudo-code gần phần gộp theo file: main.tex:125–129 hoặc trước Results main.tex:243.
  - [ ] Hoàn thiện Bảng Ablation: main.tex:299–323 — điền số liệu hoặc đánh dấu “—”; ghi chú trường hợp single-class.
  - [ ] Thêm bảng hiệu năng và tiểu mục ngắn (Complexity): gần Experimental Setup main.tex:167 hoặc thêm tiểu mục mới.
  - [ ] Bỏ chú thích tiếng Việt trong TikZ và thống nhất chú giải tiếng Anh: ví dụ main.tex:201–212, 222–231.
  - [ ] Thay câu “will be released upon acceptance” bằng liên kết ẩn danh: main.tex:365.

- refs.bib
  - [ ] Kiểm tra mục có năm “in-press” (vd: Information Fusion 2025) và DOI; chỉnh urldate (tránh ngày tương lai), ví dụ refs.bib:204–212.

- Mã/config
  - [ ] Đảm bảo file cấu hình được tham chiếu có tồn tại và là bản cuối: configs/logs_stft_full_temporal_gating.yaml (được nhắc ở main.tex:167).
  - [ ] Tham số hóa ngưỡng stage trong datasets/logs_ttf.py; mở cờ CLI và log vào metadata của run.
  - [ ] Thêm script: train_dev.sh/.ps1, train_temporal.sh/.ps1, eval_temporal.sh/.ps1 với vòng lặp seed.

- Hình
  - [ ] Xác minh mọi đường dẫn hình (đã kiểm: figures/temporal/*, figures/fullrange/*). Thêm PR/ROC/calibration còn thiếu.

Gói Nộp Bài — Checklist
- [ ] Tuân theo giới hạn sn-jnl của Springer Nature; đã bật microtype/xurl.
- [ ] PDF biên dịch không có overfull box/cảnh báo; dọn file phụ trợ.
- [ ] Thư cover: nhấn mạnh đánh giá tránh rò rỉ, lợi ích đa cảm biến, định hướng triển khai.
- [ ] Code ẩn danh/tài liệu bổ sung đã tải; README kèm lệnh chạy chính xác.
- [ ] Tuyên bố dữ liệu trỏ DOI/URL ổn định; nêu rõ subset, RUNS/FILES.

Lộ Trình Gợi Ý
- Tuần 1: Baseline (1D CNN, ResNet18), định thức present-class, dọn bảng ablation.
- Tuần 2: Cross-run/bộ ngoài, chạy đa seed, đo hiệu năng.
- Tuần 3: Hình PR/ROC, calibration, mở rộng Related Work, rà refs.
- Tuần 4: Repo tái lập (ẩn danh), chỉnh sửa cuối, biên dịch và nộp.

Phân Công (điền)
- Thí nghiệm: [...]
- Viết: [...]
- Mã/phát hành: [...]
- Hình: [...]
