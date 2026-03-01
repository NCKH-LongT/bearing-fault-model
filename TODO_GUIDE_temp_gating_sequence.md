# Hướng dẫn nhanh: cải thiện nhánh nhiệt và thêm yếu tố thời gian (để lấy số liệu nhanh)

Mục tiêu: cải thiện F1 Degrading ở lát cắt [70,90]% mà không làm giảm mạnh Fault ở [90,100.1]%, bằng các chỉnh sửa tối thiểu: tiền xử lý nhiệt tốt hơn + hợp nhất có kiểm soát (gating, modality dropout). Tùy chọn bước 2: thêm GRU nhẹ cho chuỗi.

## 1) Tạo config mới (bản sao + tham số tối thiểu)

- Sao chép: `configs/logs_stft_full_temporal.yaml` → `configs/logs_stft_full_temporal_gating.yaml`
- Bổ sung các khóa sau (ví dụ):

```yaml
# configs/logs_stft_full_temporal_gating.yaml
experiment_name: full_temporal_gating

features:
  temp:
    delta: true                # dùng deltaT = bearing - ambient
    zscore_per_file: true      # z-score theo từng file
    ewma_sec: [10, 30]         # làm trơn/EWMA đa thang (giây)
    clip_percentile: 0.995     # cắt outlier nhẹ (tùy chọn)

fusion:
  gating: true                 # bật gating theo độ tin cậy nhiệt
  modality_dropout_p: 0.3      # tắt ngẫu nhiên kênh nhiệt khi train

eval:
  slices: [[70, 90], [90, 100.1], [70, 100.1]]  # các lát cắt cần báo cáo
  present_labels: true                          # tính metric trên lớp hiện diện
```

Giữ nguyên phần tối ưu/AMP/early-stop như cấu hình hiện tại (đã có trong Bảng 1 của bài).

## 2) Tiền xử lý nhiệt (deltaT + z-score theo file + EWMA)

Nơi chỉnh: pipeline sinh 6-D nhiệt (ví dụ trong loader hoặc module đặc trưng). Ý tưởng:

- deltaT = temperature_bearing − temperature_ambient
- z-score theo file: dùng mean/std trên toàn file (khi train thì tính trên train; khi eval thì cố định thống kê từ train nếu có, hoặc z-score per-file ở chế độ eval để đơn giản trước mắt)
- EWMA đa thang (10–30s) cho mean/std/slope → vẫn sinh 6-D để không phá vỡ kích thước đầu vào hiện tại

Pseudo-code:

```python
def temp_6d_features(temp_bearing, temp_ambient, fs, ewma_sec=(10,30), clip_p=0.995):
    # 1) deltaT
    delta = temp_bearing - temp_ambient

    # 2) z-score per file (một pass trên toàn chuỗi file)
    mu, sigma = delta.mean(), delta.std().clamp_min(1e-6)
    delta_z = (delta - mu) / sigma

    # 3) clip outlier nhẹ (tùy chọn)
    if clip_p is not None:
        lo, hi = np.quantile(delta_z, [1-clip_p, clip_p])
        delta_z = np.clip(delta_z, lo, hi)

    # 4) EWMA đa thang
    def ewma(x, win_sec):
        win = max(1, int(win_sec*fs))
        alpha = 2/(win+1)
        y = np.zeros_like(x)
        for i, v in enumerate(x):
            y[i] = alpha*v + (1-alpha)*(y[i-1] if i>0 else v)
        return y

    feats = []
    for win_s in ewma_sec:
        z_s = ewma(delta_z, win_s)
        # tính mean, std, slope trong cửa sổ 1s hiện có (khớp với khung 1s)
        mean = z_s.mean()    # hoặc mean trong khung
        std  = z_s.std()     # hoặc std trong khung
        # slope: fit tuyến tính trong khung 1s (hoặc sử dụng vi phân)
        slope = np.polyfit(np.arange(z_s.size), z_s, 1)[0]
        feats.extend([mean, std, slope])
    # Kết quả 6-D (2 thang * 3 đặc trưng)
    return np.array(feats, dtype=np.float32)
```

Lưu ý: Trên thực tế, bạn đã có đường sinh 6-D; hãy thay nội dung tính toán bằng biến thể có deltaT + z-score + EWMA như trên, nhưng vẫn trả về 6-D để không phải thay đổi mô hình.

## 3) Hợp nhất có kiểm soát (gating + modality dropout)

Nơi chỉnh: head hợp nhất (sau khi có embedding rung và vector nhiệt 6-D). Pseudo-code PyTorch:

```python
# v_emb: (B, Dv) – embedding rung
# t_feat: (B, 6) – đặc trưng nhiệt 6-D sau tiền xử lý
# logits_v: (B, C) – logits của nhánh rung (nếu đã có)

# 0) modality dropout trong train
if self.training and torch.rand(1).item() < cfg.fusion.modality_dropout_p:
    t_feat = torch.zeros_like(t_feat)

# 1) chiếu nhiệt sang không gian phù hợp
proj_t = self.proj_t(t_feat)        # Linear(6 -> Dp), ví dụ Dp=32

# 2) gating theo độ tin cậy (alpha ∈ [0,1])
alpha  = torch.sigmoid(self.gate(t_feat))  # Linear(6 -> 1)

# 3) logits hợp nhất: cộng có trọng số
logits = logits_v + alpha * self.head_t(proj_t)   # head_t: Linear(Dp -> C)
```

Tối thiểu: thêm `self.proj_t`, `self.gate`, `self.head_t` trong `__init__`, lấy `p`/các hệ số từ config.

## 4) (Tùy chọn) Thêm GRU nhẹ cho chuỗi embedding

- DataLoader gom chuỗi các khung liên tiếp per-file: `seq_len ≈ 32–64`, `seq_stride ≈ 16`.
- Module: `GRU(Dv, 128, num_layers=1)` → `Linear(128 -> C)`; pooling cuối chuỗi (mean/last) rồi tiếp tục file-wise mean logits như hiện tại.
- YAML gợi ý:

```yaml
sequence:
  enable: true
  model: gru
  seq_len: 64
  seq_stride: 16
```

## 5) Chạy & báo cáo

- Huấn luyện temporal: sử dụng config mới

```bash
# ví dụ (tuỳ theo entrypoint của bạn)
python train.py --config configs/logs_stft_full_temporal_gating.yaml
```

- Đánh giá theo 3 lát cắt: [70,90] (Degrading), [90,100.1] (Fault), [70,100.1] (tổng hợp). Dùng `present_labels: true` nếu có lớp vắng mặt.
- Tiêu chí thành công ban đầu:
  - F1 Degrading trên [70,90]% tăng so với baseline `vibration-only`.
  - Fault F1 trên [90,100.1]% không giảm mạnh.

## 6) Cập nhật bảng/kết quả trong bài

- Thêm một dòng vào `tab:abl_summary` (trong `main.tex`) cho biến thể: `+deltaT+zscore+EWMA+gating (vib+temp)`
- Nêu ngắn gọn trong phần Ablation kết luận: 
  - Gating + làm trơn nhiệt giúp giảm lệch về Fault khi lát cắt rộng [70,100.1]%;
  - F1 Degrading [70,90]% tăng; very‑late Fault vẫn giữ tốt.

## 7) Gợi ý mở rộng nếu cần thêm

- Thử nhiều `ewma_sec`: `[5, 20]`, `[10, 40]` và `modality_dropout_p∈{0.2,0.5}`.
- Thêm smoothing logits theo thời gian trước file‑wise mean (moving/median 5–11 khung).
- Bước 2: bật `sequence.enable=true` để dùng GRU khi đã ổn định pipeline nhiệt.

---

Nếu bạn muốn, mình có thể tạo sẵn file config `configs/logs_stft_full_temporal_gating.yaml` và khung lớp `GatedFusionHead` (PyTorch) để bạn chỉ việc chạy.
