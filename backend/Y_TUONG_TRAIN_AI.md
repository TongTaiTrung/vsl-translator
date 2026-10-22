# Ý Tưởng Train AI — VSL Translator (Nhận Diện Ngôn Ngữ Ký Hiệu Việt)

Tài liệu tổng hợp từ codebase hiện tại: `src/vsl_translator/`, `scripts/`, `config.py`,
`TRAINING.md`, `CHANGELOG.md`. Mục đích: một người mới đọc xong hiểu được
dữ liệu đi từ webcam thành label như thế nào, và vì sao pipeline train được
thiết kế như hiện tại.

---

## 1. Ý tưởng tổng thể

Nhận diện cử chỉ tay từ video là bài toán chuỗi thời gian: mỗi cử chỉ là một
sequence 24 frame, mỗi frame là một vector đặc trưng số. Thay vì đưa ảnh thô
vào CNN, pipeline này dùng **MediaPipe Holistic** để bóc skeleton ra trước,
rồi train một **GRU classifier** trên chuỗi skeleton. Lợi ích:

- Vector nhẹ (188 số/frame) → model nhỏ, train nhanh, inference realtime.
- Skeleton đã tách dáng người khỏi background, ánh sáng, quần áo.
- Chuẩn hóa canonical (mục 3) giúp model bất biến khoảng cách camera và tỉ lệ cơ thể.

Pipeline đầy đủ:

```
webcam → MediaPipe Holistic → extract() → vector 188-dim
       → collect_data.py → data/<action>/<seq>/0..23.npy
       → train.py (split → augment → standardize → GRU) → model.keras
       → predict_live.py / realtime_predict.py (sliding window + majority vote)
```

---

## 2. Cách trích xuất đặc trưng (landmarks.py)

### 2.1. Detect

`detect(image, model)` (`landmarks.py:25`):

1. BGR → RGB, đánh dấu `writeable=False` để MediaPipe xử lý nhanh.
2. `model.process()` với `model_complexity=2`, `smooth_landmarks=True`,
   ngưỡng detection/tracking thấp (0.01) để ít miss tay khi che khuất.
3. `finally` khôi phục `writeable=True` kể cả khi exception.

Output: `results` chứa `pose_landmarks` (33 điểm), `left/right_hand_landmarks`
(mỗi bên 21 điểm), `face_landmarks` (468 điểm — chỉ dùng 2 mắt ngoài 33/263
làm fallback tâm mặt khi mất tai pose).

### 2.2. Neo vai và kiểm tra tin cậy

- Chỉ dùng 6 khớp pose: vai (11, 12), khuỷu (13, 14), cổ tay (15, 16).
- `_get_shoulders()`: cả hai vai phải tồn tại và
  `visibility >= SHOULDER_VIS_THRESHOLD (0.5)`, ngược lại cả frame = vector 0
  để lớp `Masking(0.0)` của model bỏ qua.
- `shoulder_width()` dùng `|x_ls − x_rs|` (trục X thuần), không dùng hypot 2D —
  ổn định khi nghiêng người, không phụ thuộc aspect ratio khung hình.

### 2.3. Canonical Skeleton Retargeting (Forward Kinematics)

Ý tưởng cốt lõi: **giữ hướng xương, chuẩn hóa độ dài xương**.
Vai ghim tại `(∓0.5, 0)`, khuỷu và cổ tay dựng lại từ unit vector của xương thật:

```
elbow = shoulder_canon + unit(elbow_raw − shoulder_raw) × 1.2
wrist = elbow + unit(wrist_raw − elbow_raw) × 1.0
```

Hệ quả: người gần/xa camera, tay dài/ngắn khác nhau vẫn cho cùng vector nếu
cùng tư thế. Tay đứng yên cho giá trị ổn định khác 0 (không bị triệt tiêu như
khi trừ trung điểm vai).

### 2.4. Chuẩn hóa bàn tay theo palm_size

`_process_hand(hand_lm)`: dịch 21 điểm tay về cổ tay của chính hand-detector
(`pts[0]`), chia cho `palm_size` riêng của bàn tay đó:

```
palm_size = max(|w0 − mcp9|, |mcp5 − mcp17| × 1.25)
normed = (pts − w0) / palm_size × 0.4
```

Nhánh `max` giữ scale ổn định khi bàn tay foreshorten về phía camera (một trục
sập, trục còn lại sống). Không cộng cổ tay canonical vào slot tay — vị trí toàn
cục đã nằm ở slot pose. Tay mất hoặc palm suy biến (`< 0.012`) → `zeros(42)`.

### 2.5. Kênh arm_depth (độ sâu đẩy/kéo tay)

Tín hiệu foreshorten 2D không phân biệt được "đẩy tay về camera" với "gập khuỷu
trong mặt phẳng ảnh", nên có 2 kênh riêng `[102:104]`:

```
ratio = (upper_len + fore_len pixel) / neutral_len + 2.0 × (wrist.z − shoulder.z)
clip [0.3, 2.0], < 1.0 = đẩy về camera
```

- Đo bằng **pixel** (`frame_shape=(h, w)`), không đo trên tọa độ normalized —
  tránh méo aspect ratio.
- `neutral_len` là độ dài tay trung tính **hiệu chuẩn theo từng phiên**
  (`ArmDepthCalibrator`: median ~20 frame non-idle, reject ngoài 1.5–4.0×
  shoulder width), thay cho hằng số 2.2 cũ.
- `dz` từ MediaPipe chỉ là hiệu chỉnh phụ (nhiễu, tương đối).
- `ArmDepthSmoother` (EMA alpha 0.4 + median window 5) chỉ dùng ở realtime,
  data train giữ raw.
- Idle-gate: tay buông thõng (cổ tay canonical y > 1.75 và |x| < 0.95) → zero
  phía đó, nhưng depth **giữ giá trị đo** (không ép 1.0 như bản cũ gây bước nhảy).

### 2.6. Layout vector 188-dim (BODY_ONLY_DIM)

| Đoạn | Nội dung | Index |
|---|---|---|
| Pose trái | vai, khuỷu, cổ tay × (x, y, vis) | [0:9] |
| Pose phải | vai, khuỷu, cổ tay × (x, y, vis) | [9:18] |
| Tay trái | 21 điểm × (x, y) local | [18:60] |
| Tay phải | 21 điểm × (x, y) local | [60:102] |
| Arm depth | ratio trái, phải | [102:104] |
| Face-offset trái | 21 điểm × (x, y) so với tâm mặt | [104:146] |
| Face-offset phải | 21 điểm × (x, y) so với tâm mặt | [146:188] |

Tâm mặt = trung điểm 2 tai pose (landmark 7, 8); fallback 2 mắt ngoài face
mesh (33, 263). Mỗi offset = `(điểm_tay − tâm_mặt) / palm_size × 0.4` — cùng
hệ canonical với slot tay, bất biến khoảng cách camera. Thiếu mặt → block 0;
idle-gate zero face-offset cùng bên tay idle.

`0.0` = missing (vai mất / tay ngoài khung / frame drop). Fallback theo từng
bên: hỏng một bên thì bên kia giữ nguyên.

### 2.7. Không còn motion features (đã bỏ)

Các kênh motion (vel, speed, displacement, extension, d_arm_depth, 14 kênh)
đã bị xóa: GRU tự học động lực thời gian từ chuỗi 188-dim thô, không cần
đạo hàm viết tay. `n_features()` trả 188.

---

## 3. Cách hoạt động của training (train.py + dataset.py + model.py)

### 3.1. Dữ liệu

`scripts/collect_data.py` thu mỗi action 15 sequence (`N_SEQUENCES`), mỗi
sequence 24 frame (`SEQUENCE_LEN`), lưu `data/<action>/<seq_id>/0..23.npy`.
`load_raw_dataset()` đọc thành `X (N, 24, 188)`, `y` one-hot. Bước này không
augment, không chuẩn hóa.

### 3.2. Thứ tự chống rò rỉ (bắt buộc)

```
raw → split test (15%, stratify, seed 42) → split train/val (15% tổng)
    → augment CHỈ train (×8) → standardize bằng stats train → fit
```

- Test held-out không augment, không early-stopping, không chọn checkpoint —
  đây là con số chính. Val hơi lạc quan vì checkpoint chọn theo val.
- Chuẩn hóa: mean/std tính trên train-augment, bỏ qua entry 0.0
  (nanmean/nanstd), rồi khôi phục 0.0 sau chuẩn hóa. Stats lưu
  `models/feature_stats.npz` để inference dùng chung.
- Bài học xương máu (CHANGELOG): từng để bản sao `add_noise.py` chung trong
  `data/` và augment trước split → bản gốc và bản copy rơi vào hai phía
  train/val, accuracy ảo cao. Quy tắc: không chạy `add_noise.py` vào `data/`
  trước khi train.

### 3.3. Augment (train-only, dataset.py)

Thứ tự trong `augment_sequence()`: noise → scale-jitter (50%) → time-warp
(50%) → shift 2D (50%) → frame-drop (30%) → hand-dropout → hand-swap →
inactive-injection. Nguyên tắc chung: mọi augment cộng tính đều **né kênh
visibility và kênh ratio** (`_coord_mask()`), entry 0.0 giữ nguyên 0.0.

| Augment | Tham số | Mục đích |
|---|---|---|
| Gaussian noise | std 0.05, chỉ entry ≠ 0 | jitter đo landmark |
| Scale jitter | ~N(1, 0.05) per-frame | mô phỏng nhiễu đo rộng vai |
| Time warp | sigma 0.2, mask < 0.99 → 0 | nhanh/chậm nhịp cử chỉ, vis lấy nearest-frame |
| Shift 2D | (sx, sy) uniform ±0.02 | jitter gốc tọa độ |
| Frame drop | p 0.05 | mô phỏng tracking mất frame |
| Hand dropout | p 0.5, bỏ qua mẫu 1 tay | phá shortcut "tay nào hiện" |
| Hand swap | p 0.3, bỏ qua mẫu 1 tay | ép bất biến phía tay, học quỹ đạo |
| Inactive injection | p 0.35, cổ tay y 1.1–1.6 + drift 0.015 | sinh tay nghỉ giả cho mẫu 1 tay, ép học tay active |

Ba augment cuối sinh ra từ các lần debug bias thật: model từng học shortcut
"khối tay nào khác 0" thay vì động tác (hello thu tay phải, nhanvien thu tay
trái), và từng nhầm 1-tay thành 2-tay ở xa do feature tay bùng biên độ.

### 3.4. Model (model.py)

`build_model(n_actions)`: `Masking(0.0)` → BiGRU-64 (return sequences) →
LayerNorm + Dropout 0.4 → BiGRU-64 → LayerNorm + Dropout 0.4 → Dense 128 →
LayerNorm + Dropout 0.4 → Dense 64 → LayerNorm + Dropout 0.3 → softmax.
Nhỏ một cách có chủ đích: dataset ~90 mẫu raw không nuôi nổi stack sâu.

- `LayerNormalization` thay `BatchNormalization` (BN nhiễm frame mask 0 và
  moving-average nhiễu trên dataset nhỏ).
- Loss `CategoricalFocalCrossentropy(alpha=1.0, gamma=2.0, smoothing=0.05)`;
  alpha scalar không cân bằng class — cân bằng do `class_weight` tính trên
  nhãn train đã augment.
- Adam `lr=1e-3`, clipnorm 1.0 chống exploding gradient.

### 3.5. Callbacks và dừng train

- `CosineAnnealingWarmup`: warmup tuyến tính 10 epoch rồi cosine về 1e-6.
- `SmartEarlyStopping`: patience 60 từ epoch 30; dừng khi đạt target
  (val_acc 0.98 + val_loss 0.05), val_loss tăng 15 epoch liên tiếp, plateau 30
  epoch, hoặc gap train-val vượt 0.25 suốt 30 epoch.
- `ModelCheckpoint` lưu best theo `val_loss`; backup model cũ (giữ tối đa 5).
- Đánh giá cuối trên checkpoint tốt nhất: train-aug (tham khảo fit), train-clean
  (raw train chuẩn hóa bằng train stats — gap với val phản ánh khái quát hóa
  thật), val (tuning signal), held-out test (kết quả chính) + confusion matrix.

---

## 4. Inference realtime

`predict_live.py` / `realtime_predict.py`: sliding window 24 frame vector 188-dim
trực tiếp từ `extract()` → `model.predict` → ngưỡng
`CONFIDENCE_THRESHOLD = 0.70` (dưới ngưỡng = uncertain) → majority vote trên
`SMOOTHING_WINDOW = 5` dự đoán gần nhất. Mất tay ≥ 5 frame → clear buffer.
Mỗi session giữ 1 `ArmDepthCalibrator` + 1 `ArmDepthSmoother` riêng.

---

## 5. Ý tưởng cải tiến (mở, chưa làm)

1. **Thu data đa khoảng cách thật (2–3 khoảng cách/action):** scale đồng nhất
   trong augment không mô phỏng được phối cảnh thật và tracking kém ở xa.
2. **Thêm class phân biệt bằng tay trái/phải:** hiện tại hand-swap ép bất biến
   phía tay; nếu có class lateralized phải tắt swap cho class đó.
3. **Depth từ stereo/thời gian:** `dz` MediaPipe nhiễu; có thể học depth từ
   chuỗi foreshorten thay vì hiệu chỉnh tuyến tính.
4. **Tăng AUGMENT_FACTOR hoặc thu thêm mẫu** khi thêm action mới — 15 seq/action
   là mức tối thiểu, model GRU nhỏ sẽ bão hòa khi số class tăng.
5. **Contrastive pretrain trên skeleton:** dùng dữ liệu chưa gán nhãn để học
   embedding cử chỉ trước, rồi fine-tune classifier — hợp khi data nhãn ít.
