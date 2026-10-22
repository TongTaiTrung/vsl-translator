# Huấn luyện mô hình VSL (Training Guide)

Tài liệu này mô tả chính xác `scripts/train.py` làm gì: dữ liệu `.npy` được gom
thành mẫu train ra sao, pipeline chống rò rỉ, và từng giai đoạn huấn luyện.

## 1. Cấu trúc dữ liệu đầu vào

```
data/
├── hello/
│   ├── 1/
│   │   ├── 0.npy   # frame 0: vector 188 số (float32)
│   │   ├── 1.npy
│   │   └── ...
│   │   └── 23.npy  # frame 23 (SEQUENCE_LEN = 24)
│   ├── 2/
│   └── ...
├── nhanvien/
└── ...
```

- Mỗi **sequence** = 1 thư mục con (`data/<action>/<seq_id>/`) chứa đúng 24 file
  `0.npy … 23.npy`.
- Mỗi **frame** = 1 vector 188-dim (float32), layout:
  pose 6 khớp × (x, y, vis) = 18 số `[0:18]` (per-side: trái `[0:9]`,
  phải `[9:18]`, mỗi bên vai/khuỷu/cổ tay) + tay trái 42 số `[18:60]` +
  tay phải 42 số `[60:102]` + arm depth 2 số `[102:104]` +
  face-offset trái 42 số `[104:146]` + face-offset phải 42 số `[146:188]`
  (mỗi điểm tay so với tâm mặt, chuẩn hóa palm_size — xem Y_TUONG_TRAIN_AI.md §2.6).
- Vai canonical cố định tại `(∓0.5, 0)`; khuỷu/cổ tay dựng lại bằng Forward
  Kinematics từ **hướng xương** (unit vector) với độ dài chuẩn 1.2/1.0 — bất
  biến với dài tay, rộng vai và khoảng cách camera, tay đứng yên vẫn khác 0.
  Mỗi bàn tay chuẩn hóa bằng `palm_size` riêng và neo tại cổ tay canonical.
- Giá trị `0.0` = keypoint missing (vai mất / tay ngoài khung / frame drop).
- `data/` **chỉ chứa sequence thu tay**. Bản sao do `scripts/add_noise.py`
  sinh ra không được để trong `data/` khi train (đã từng gây rò rỉ, xem mục 3).

## 2. Cách gom `.npy` thành mẫu train (`load_raw_dataset`)

Hàm `load_raw_dataset()` trong `src/vsl_translator/dataset.py`:

1. Duyệt từng action trong `cfg.ACTIONS`, liệt kê thư mục sequence con
   (sắp xếp theo số).
2. Với mỗi sequence, đọc lần lượt `0.npy … 23.npy`. Thiếu bất kỳ frame nào
   → bỏ cả sequence (`valid_seq = False`).
3. Xếp 24 frame thành mảng `(24, 188)`, gán nhãn `label_map[action]`.
4. Kết quả: `X` shape `(N, 24, 188)`, `y` one-hot shape `(N, n_classes)`.

Không có augment, không có chuẩn hóa ở bước này — dữ liệu raw nguyên vẹn.

## 3. Pipeline chống rò rỉ (thứ tự bắt buộc)

```
raw → split 3-way → augment CHỈ train → standardize bằng stats của train → fit
```

1. **Split test trước**: `train_test_split(test_size=TEST_SPLIT=0.15,
   stratify)` tách held-out test. Test không bao giờ augment, không dùng cho
   early stopping hay chọn checkpoint.
2. **Split train/val**: trên phần còn lại, tỉ lệ val tính lại
   `VAL_SPLIT / (1 − TEST_SPLIT)` để val đúng 15% tổng. Cả hai split đều
   `stratify` + `RANDOM_SEED=42`.
3. **Lưu test**: `models/test_set.npz` (X, y raw) để `predict.py` đánh giá sau.
4. **Augment chỉ train**: `augment_train_data(X_train_raw, AUGMENT_FACTOR=8)` —
   giữ bản gốc + 8 bản biến đổi cho mỗi mẫu (noise, scale-jitter, time-warp,
   shift 2D, frame-drop, hand-dropout 0.5, hand-swap 0.3). Val giữ nguyên.
5. **Standardize**: `standardize_features` tính mean/std **trên train-augment**,
   bỏ qua entry 0.0 (tính bằng nanmean/nanstd), rồi khôi phục 0.0 sau chuẩn hóa.
   Val/test chuẩn hóa bằng đúng mean/std này. Stats lưu vào
   `models/feature_stats.npz` để inference dùng chung.
6. **Vì sao thứ tự này quan trọng**: nếu augment trước split, hoặc để bản sao
   `add_noise.py` nằm chung trong `data/`, thì bản gốc và bản copy của cùng một
   gesture rơi vào hai phía train/val — val/test bị nhiễm, accuracy ảo cao.

## 4. Mô hình và huấn luyện

- Kiến trúc (`src/vsl_translator/model.py`): `Masking(0.0)` → BiLSTM-64 ×2
  (LayerNorm + Dropout 0.4) → Dense 128/64 (L2 1e-4, LayerNorm, Dropout) →
  softmax. Loss `CategoricalFocalCrossentropy(alpha=1.0, gamma=2.0,
  smoothing=0.05)`, Adam `lr=1e-3` + clipnorm 1.0.
- Class weight tính trên **nhãn train đã augment** (đúng phân phối optimizer thấy).
- Callbacks: `CosineAnnealingWarmup` (warmup 10 epoch rồi cosine về 1e-6),
  `TrainingMonitor` (bảng log mỗi epoch + cảnh báo overfit gap > 0.25),
  `SmartEarlyStopping` (patience 60 từ epoch 30, dừng khi đạt target
  val_acc 0.98 + val_loss 0.05, hoặc val_loss tăng 15 epoch liên tiếp, hoặc
  plateau 30 epoch, hoặc gap overfit 30 epoch), `ModelCheckpoint` lưu best theo
  `val_loss`, TensorBoard ra `logs/`.
- Backup model cũ vào `models/backups/` (giữ tối đa 5 bản) trước khi train.

## 5. Đánh giá cuối

1. Nạp lại checkpoint tốt nhất từ đĩa rồi đánh giá 3 con số:
   - **train-aug**: độ fit trên bản nhiễu (chỉ để tham khảo).
   - **train-clean**: raw train chuẩn hóa bằng train stats — gap
     `train-clean − val` mới phản ánh khái quát hóa thật.
   - **val**: tuning signal, hơi lạc quan vì checkpoint đã chọn theo val.
2. In classification report + confusion matrix trên val.
3. **Held-out test** là kết quả chính (accuracy + report) — mẫu chưa từng tham
   gia train/val/augment/early-stopping.
4. Phân tích confidence trên val (mean/min/max, đếm dự đoán < 70%).

## 6. Chạy train

```bash
venv/bin/python scripts/train.py
```

Kết quả: `models/model.keras`, `models/feature_stats.npz`,
`models/test_set.npz`, log TensorBoard trong `logs/`. Xem tiến trình:

```bash
tensorboard --logdir logs
```
