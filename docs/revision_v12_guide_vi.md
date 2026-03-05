# Hướng dẫn Chỉnh sửa Bài báo Khoa học: Nâng cấp từ Version 11 lên Version 12

Tài liệu này tổng hợp các thay đổi đề xuất để nâng cấp bản thảo từ Version 11 lên Version 12, nhấn mạnh giao thức đánh giá leakage-aware, căn chỉnh theo thời gian còn lại đến hỏng (TTF-aligned) và tối ưu hoá pipeline đa phương thức cho triển khai thực tế.

## 1) Tổng quan về Mục tiêu Chỉnh sửa
- Chuyển từ báo cáo kết quả thực nghiệm thuần tuý sang thiết lập khung đánh giá leakage-aware, tôn trọng trình tự thời gian (chronology) và tránh rò rỉ cửa sổ chồng lấn.
- Tập trung vào tính thực tiễn và độ tin cậy trong kịch bản run-to-failure, với suy luận theo tệp (file-wise) bằng mean-logit aggregation.
- Kết quả tiêu biểu (leakage-aware): Accuracy toàn vòng đời 0.8915, Macro-F1 0.8739 (majority vote), Fault F1 0.9600 ở giai đoạn cuối đời.

Các cải thiện trọng tâm (05 điểm):
- Introduction: Nhấn mạnh rò rỉ dữ liệu thời gian và khó khăn của giai đoạn thoái hoá sớm.
- Related Works (SOTA): Làm rõ ưu thế giao thức TTF-aligned so với chia ngẫu nhiên mức cửa sổ.
- ResNet-2D Encoder: Diễn giải kiến trúc CNN 2D nhỏ gọn cho edge deployment.
- TTF Thresholds: Giải trình các ngưỡng phân chia sức khỏe và cơ chế padding kỹ thuật.
- Consolidated Ablation (Table 3): Bảng đối sánh toàn diện, làm rõ sự sụp đổ của phương pháp truyền thống ở late-life.

## 2) Hiệu chỉnh Chương 1: Introduction (Mở đầu)
Hai thách thức cốt yếu được giải quyết:
- Nhận diện giai đoạn Degrading mờ nhạt và mất cân bằng.
- Thiết lập quy trình đánh giá leakage-aware để phản ánh đúng hiệu suất triển khai, tránh lạc quan ảo do chia ngẫu nhiên.

Chỉ số thực nghiệm tiêu biểu:
- Accuracy full-range: 0.8915; Macro-F1: 0.8739 (majority vote).
- Fault F1 (late-life): 0.9600.

Bảng 1: Thách thức và giải pháp đề xuất

| Thách thức (Challenge) | Giải pháp của bài báo (Proposed Solution) |
| --- | --- |
| Kết quả lạc quan ảo do rò rỉ dữ liệu ở mức cửa sổ (window-level leakage). | Giao thức đánh giá nhận biết rò rỉ, căn chỉnh theo thời gian còn lại (TTF-aligned). |
| Giai đoạn thoái hóa khó phân biệt với trạng thái bình thường. | Hợp nhất đa phương thức (Vibration + Temperature) để khai thác xu hướng bổ trợ. |
| Độ sáng ảnh phổ STFT không ổn định giữa các cửa sổ thời gian. | Chuẩn hoá theo từng tần số (per-frequency normalization) để ổn định đặc trưng. |

## 3) Hiệu chỉnh Chương 2: Related Works (Nghiên cứu liên quan)
- Giao thức đánh giá: Nhiều nghiên cứu trước dùng random/window-level splits khiến các cửa sổ chồng lấn xuất hiện ở cả train và test. Giao thức "Leakage-aware, TTF-aligned" của chúng tôi tôn trọng chronology, giúp mô hình học tổng quát hoá từ quá khứ → tương lai (như minh hoạ Fig. 2).
- Giá trị đa phương thức: Khác với chỉ-rung, tích hợp nhiệt độ tăng nhạy ở giai đoạn bắt đầu thoái hoá. Nhiệt phản ánh tổng hợp tải và ma sát, nổi bật ở late-life dù rung có tính non-stationary cao.

## 4) Hiệu chỉnh Chương 3: Methodology (Phương pháp)
- STFT: `n_fft = 2048`, `hop_length = 512` (Hann), resize ảnh phổ về 160×160, chuẩn hoá theo từng tần số (per-frequency normalization) dọc trục thời gian.
- ResNet-2D Encoder: CNN 2D nhỏ gọn mã hoá 2 kênh rung thành embedding 256 chiều (R^256), tối ưu tính toán biên (ví dụ RTX 2060).
- Luồng xử lý (Fig. 1):
  - Trích xuất: STFT 2 kênh cho rung; đặc trưng nhiệt 6D (mean, std, slope cho 2 kênh).
  - Late Fusion: Nối embedding rung (R^256) với embedding nhiệt (R^32), qua linear head để phân loại.
  - File-wise inference: Mean-logit aggregation trên tất cả cửa sổ của tệp, sau đó argmax; chạy AMP để tối ưu GPU.

## 5) Hiệu chỉnh Chương 4–5: Experimental Setup & Results
Giải trình ngưỡng TTF (Time-to-Failure):

| Giai đoạn (Stage) | Ngưỡng TTF (%) | Giải trình thực nghiệm |
| --- | --- | --- |
| Healthy | [0, 0.60) | Vận hành ổn định, tín hiệu nền. |
| Degrading | [0.60, 0.90) | Dấu hiệu thoái hoá sớm, nhiệt có xu hướng tăng. |
| Fault | (0.90, 1.00] | Hỏng hóc nghiêm trọng. Lưu ý: ngưỡng 100.1% là padding kỹ thuật để bao phủ dữ liệu tệp cuối. |

Failure Case Analysis:
- Ranh giới Healthy–Degrading (60–70% TTF) khó do chuyển đổi dần (gradual). Trên lát rộng [70, 100.1]%, Accuracy = 0.7949 và Macro-F1 = 0.4790, thể hiện nhầm lẫn đáng kể giữa Degrading và Healthy vì chồng lấn và support hạn chế ở biên (xem Fig. 3, Fig. 4).

## 6) Table 3: Consolidated Ablation Summary
Sử dụng "present-class scoring" để đánh giá chính xác hiệu suất trên lát thời gian khi có lớp vắng mặt.

| Slice (%) | Variant | Aggregation | Accuracy | Macro-F1 (present) | F1 Degrading | F1 Fault |
| --- | --- | --- | --- | --- | --- | --- |
| 70, 90 | Multi-modal (vib+temp) | Mean-logit | 0.8846 | 0.9388 | 0.9388 | — |
| 70, 90 | Classical vib (SVM, 8D) | — | 1.0000 | 1.0000 | 1.0000 | — |
| 85, 100.1 | Vib-only | Mean-logit | 0.2593 | 0.5185 | 0.5185 | 0.0000 |
| 85, 100.1 | Multi-modal (vib+temp) | Mean-logit | 0.9467 | 0.9333 | — | 0.9600 |
| 70, 100.1 | Vib-only | Mean-logit | 0.8000 | 0.8000 | 0.8000 | 0.0000 |
| 70, 100.1 | Multi-modal (vib+temp) | Mean-logit | 0.8974 | 0.9045 | 0.9200 | 0.8889 |
| 90, 100.1 | Multi-modal (vib+temp) | Mean-logit | 0.9231 | 0.9600 | — | 0.9600 |
| 90, 100.1 | Classical vib (SVM, 8D) | — | 0.0000 | 0.0000 | — | 0.0000 |
| 90, 100.1 | Vib-only | Mean-logit | 0.0000 | 0.0000 | — | 0.0000 |

Ghi chú: SVM 8D có thể tốt ở sớm (ví dụ [70, 90]%) nhưng sụp đổ ở giai đoạn Fault, khẳng định lợi ích rõ rệt của đa phương thức.

## 7) Kết luận và Hướng phát triển
- Kết hợp nhiệt độ cải thiện mạnh nhận diện Fault ở late-life (F1 = 0.9600). Giao thức leakage-aware TTF-aligned loại bỏ lạc quan ảo, phản ánh khả năng triển khai.
- Hướng phát triển:
  - Đánh giá cross-bearing: mở rộng sang CWRU và Paderborn để kiểm chứng tổng quát hoá liên miền.
  - Phân tích độ nhạy ngưỡng: xem xét tinh chỉnh mốc 60% và 90% TTF cho cảnh báo sớm.
  - Mô hình hoá chuỗi (Sequence-aware): thử Transformer/RNN để tận dụng ngữ cảnh nhiệt dài, giảm nhầm lẫn Healthy–Degrading.

## 8) Kiểm soát Chất lượng (Final Checklist)
- Thuật ngữ: STFT, ResNet-2D, TTF-aligned, leakage-aware, mean-logit aggregation.
- Hình ảnh: Gọi Hình 1 (Pipeline), Hình 2 (Protocol), Hình 3 (Late-life confusion), Hình 4 (Full-range analysis).
- Dữ liệu: Bổ sung Macro-F1 = 0.8739; nêu rõ sự sụp đổ của SVM ở Table 3.
- Kỹ thuật: Nêu các thông số reproducibility (n_fft, hop_length, GPU, AMP).

