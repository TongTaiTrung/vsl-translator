# vsl2

Module độc lập cho nhận diện ngôn ngữ ký hiệu (VSL): **train, thu thập dữ
liệu, ingest video, tăng cường nhiễu, quản lý class và inference** — tất cả
gói gọn trong một thư mục, không import bất kỳ code nào khác trong dự án.

Phong cách sử dụng mượn ý tưởng từ `ultralytics`: một class `VSL2` duy nhất,
gọi `.train()` / `.predict()` là chạy. Nhưng **không dùng package
ultralytics** — framework train là TensorFlow/Keras (BiGRU + attention
`WeightedMean`, focal loss, masking).

## Mục lục

1. [Cài đặt](#cài-đặt)
2. [Dùng nhanh](#dùng-nhanh-python-api)
3. [CLI](#cli)
4. [Workflow điển hình](#workflow-điển-hình-thu-data--train)
5. [Cấu trúc module](#cấu-trúc-module)
6. [Chi tiết từng lệnh](#chi-tiết-từng-lệnh)
7. [Config](#config-defaultspy)
8. [Định dạng dữ liệu](#định-dạng-dữ-liệu)
9. [Quy tắc khi sửa module](#quy-tắc-khi-sửa-module-này)

## Cài đặt

```bash
venv/bin/pip install -r vsl2/requirements.txt
```

Gồm: `tensorflow==2.18.1`, `opencv-python`, `mediapipe==0.10.21`,
`scikit-learn`, `numpy`, `scipy` (pin theo root `requirements.txt`; đã loại
`openai/gtts/SpeechRecognition/PyAudio` vì đó là TTS/STT của backend, ngoài
phạm vi Core ML).

Mọi đường dẫn mặc định đều nằm trong `vsl2/` (`data/`, `weights/`,
`actions.json`). Chạy mọi lệnh từ repo root.

Checkpoint trong `vsl2/weights/` (`model.keras`, `feature_stats.npz`,
`train_record.json`, `test_set.npz`) được copy từ `backend/models/` —
cùng 8 class trong `actions.json`. Không train lại trong module này;
muốn cập nhật model thì train ở backend rồi copy 4 file sang.

## Dùng nhanh (Python API)

```python
from vsl2 import VSL2

# --- Train (override config qua kwargs, tên trùng defaults.py, hoa/thường đều được)
m = VSL2()
m.train(data="vsl2/data", epochs=50, augment_factor=8)

# --- Batch predict: file video, thư mục ảnh, thư mục seq, hoặc 12 vector đặc trưng
out = VSL2("vsl2/weights/model.keras").predict("video.mp4")
print(out["raw"], out["probs"])   # vd: "xinchao" [0.01, 0.95, ...]
out = VSL2("vsl2/weights/model.keras").predict("vsl2/data/xinchao/1")  # seq dir 0..11.npy
vecs = [np.load(f"vsl2/data/xinchao/1/{i}.npy") for i in range(12)]
out = VSL2("vsl2/weights/model.keras").predict(vecs)        # auto-route
out = VSL2("vsl2/weights/model.keras").predict_features(vecs)  # tường minh, bỏ qua MediaPipe

# --- Webcam realtime (nhấn q để thoát), trả về list câu đã commit
sentence = VSL2("vsl2/weights/model.keras").predict_live()

# --- Thu thập dữ liệu: webcam hoặc video có sẵn
m.collect(action="xinchao", sequences=15)          # webcam
m.ingest("clip.mp4", action="xinchao")            # video -> data/xinchao/<seq>/ (12 .npy)

# --- Sinh nhiễu / quản lý class
m.noise(action="xinchao", copies=2, noise_std=0.02)
m.classes("list")
m.classes("add", ["tenmoi"])
m.classes("delete", ["tencu"])
```

## CLI

```bash
venv/bin/python -m vsl2 train --epochs 50 [--data DIR] [--batch-size N] [--augment-factor N]
venv/bin/python -m vsl2 predict video.mp4 [--weights vsl2/weights/model.keras]
venv/bin/python -m vsl2 live [--weights ...]        # webcam realtime, q để thoát
venv/bin/python -m vsl2 collect --action xinchao [--sequences 15] [--device macbook]
venv/bin/python -m vsl2 ingest clip.mp4 --action xinchao [--device video]
venv/bin/python -m vsl2 noise --action xinchao --copies 2
venv/bin/python -m vsl2 classes list
venv/bin/python -m vsl2 classes add tenmoi1 tenmoi2
venv/bin/python -m vsl2 classes delete tencu
```

Lưu ý `noise` qua CLI chỉ nhận `--action/--copies`; các flag nâng cao
(`--noise-std`, `--scale-range LOW HIGH`, `--mirror`, `--noise-zeros`,
`--all-actions`, `--src-seqs`, `--seed`) dùng trực tiếp
`vsl2/noise.py` hoặc qua API `m.noise(..., noise_std=..., scale_range=[...],
mirror=True)` (tên kwarg = tên flag, gạch dưới thay gạch ngang).

## Workflow điển hình: thu data → train

```bash
# 1. Khai báo class mới
venv/bin/python -m vsl2 classes add tenmoi
# 2. Thu dữ liệu (webcam hoặc video có sẵn)
venv/bin/python -m vsl2 collect --action tenmoi --sequences 30
venv/bin/python -m vsl2 ingest clip.mp4 --action tenmoi
# 3. Kiểm tra số lượng
venv/bin/python -m vsl2 classes list
# 4. Train
venv/bin/python -m vsl2 train --epochs 100
# 5. Predict thử
venv/bin/python -m vsl2 predict test.mp4
```

Sau khi thêm/xóa class phải train lại (model cũ không còn hợp lệ —
`classes.py` sẽ hỏi có xóa model cũ không).

## Cấu trúc module

| File | Nguồn | Chức năng |
|---|---|---|
| `api.py` | mới | Class `VSL2`: `.train/.predict/.predict_live/.collect/.ingest/.noise/.classes` |
| `__main__.py` | mới | CLI `python -m vsl2 ...` |
| `defaults.py` | copy `backend/config.py` | Toàn bộ hằng số + path nội bộ `vsl2/`; đọc `actions.json` qua `reload_actions()` |
| `actions.json` | copy `backend/actions.json` | Danh sách class (hiện tại 8 class) |
| `landmarks.py` | copy `backend/src/vsl_translator/landmarks.py` | MediaPipe Holistic, `extract()` → vector **356-dim** (pose 18 + tay 84 + arm_depth 2 + shoulder-offset 252) |
| `dataset.py` | copy `backend/src/vsl_translator/dataset.py` | Load `.npy`, augment in-memory train-only, `standardize_features` (giữ mask 0.0) |
| `model.py` | copy `backend/src/vsl_translator/model.py` | BiGRU 2 lớp + custom layers `WeightedMean`/`Slice`/`FrameVelocity`, focal loss |
| `callbacks.py` | copy nguyên (không dính config) | Cosine warmup, `TrainingMonitor`, `SmartEarlyStopping` |
| `train.py` | copy `backend/scripts/train.py` | Pipeline anti-leakage: raw → 3-way split (test giữ lại) → augment train → train → `train_record.json` + backup model cũ |
| `collect.py` | copy `backend/scripts/collect_data.py` | Thu webcam → `data/<action>/<seq>/*.npy + meta.json` (đánh số tiếp, không ghi đè) |
| `ingest.py` | mới | Video → `data/<action>/<seq>/` (12 `.npy`), xem chi tiết dưới |
| `noise.py` | copy `backend/scripts/add_noise.py` | Sinh bản copy nhiễu Gaussian / scale khoảng cách / mirror trên đĩa |
| `classes.py` | copy `backend/scripts/manage_classes.py` | `list/add/delete` class, đếm sequences, hỏi xóa model cũ |
| `predict.py` | mới (gộp `_common` + `convenient_predict`, bỏ Groq) | `load_classifier`, `predict_raw(frames)` (đúng SEQUENCE_LEN frames), `predict(video/thư mục)` |
| `realtime.py` | mới (thay `stream_session`) | Webcam: sliding window + vote buffer + ngưỡng `threshold_for` + sentence, callback `on_word` |
| `data/ weights/ configs/` | runtime (gitignore nội dung) | Dữ liệu, checkpoint + `feature_stats.npz` + `train_record.json` + `test_set.npz`, cấu hình chạy |

**Không mang theo** (ngoài phạm vi Core ML): `server.py` (WebSocket),
`groq_client.py` (Groq cleanup), `overlay.py`, `stream_session.py`, eval nâng
cao (`kfold_eval`, `stress_test`, `profile_robustness`), `cam_test.py`,
`migrate_data.py` (one-shot legacy), `dataset_manager.py`.

## Chi tiết từng lệnh

### train — `m.train(...)` / `vsl2 train`

Pipeline giữ nguyên từ backend, chống leakage tuyệt đối:

1. Load raw `.npy` (kèm `devices` từ `meta.json` để stratify action × camera).
2. Split 3-way: test giữ lại trước (`TEST_SPLIT`), rồi train/val
   (`VAL_SPLIT`) — test lưu vào `weights/test_set.npz`, không bao giờ augment.
3. Augment **chỉ train** (`AUGMENT_FACTOR` bản copy in-memory: noise, scale
   jitter, time warp, shift, frame drop, hand dropout/swap, inactive-side
   injection; preset `strong` thêm burst + speed-cut qua `augment_preset`).
4. Chuẩn hóa bằng stats tính **trên train** (`feature_stats.npz`), mask 0.0
   được giữ nguyên (điểm missing không trôi thành giá trị).
5. Train BiGRU: cosine LR warmup, `TrainingMonitor` in bảng từng epoch,
   `SmartEarlyStopping` (target acc/loss, loss-spike, plateau, overfit gap),
   `ModelCheckpoint` theo `val_loss`, TensorBoard vào `logs/`.
6. Nạp lại checkpoint tốt nhất, đánh giá val + **held-out test** (headline),
   in classification report + confusion matrix, ghi
   `weights/train_record.json` (hyperparams, history từng epoch, metrics);
   model cũ được backup vào `weights/backups/` (giữ tối đa `MAX_BACKUPS`).

### collect — `m.collect(...)` / `vsl2 collect`

Mở webcam, hiển thị preview có vẽ landmarks, bấm thu `SEQUENCE_LEN` (12)
frames/sequence vào `data/<action>/<seq>/0..11.npy` + `meta.json`
(`device`, `src_shape`, `std_size`). Tự đánh số tiếp sau sequence lớn nhất,
không ghi đè. Thoát giữa chừng bằng `q`.

### ingest — `m.ingest(...)` / `vsl2 ingest`

Chuyển **một video** thành một sequence featurized:

1. Đọc toàn bộ frames bằng `cv2.VideoCapture` (lỗi mở file → `ValueError`).
2. Detect MediaPipe từng frame, đánh dấu frame có tay (trái hoặc phải).
3. **Trim** các frame không tay ở đầu/cuối; nếu còn lại < 12 → `ValueError`.
4. **Downsample** về đúng 12: lặp loại frame lẻ (`i & 1`), luôn giữ frame
   đầu/cuối (biên cử chỉ); pass cuối chỉ loại đúng số cần thiết, rải đều.
5. Chạy pipeline `normalize_frame → detect → feed_calibrator → extract →
   smoother` (y hệt predict) cho từng frame, lưu `0..11.npy` + `meta.json`
   (thêm `source` = tên video) vào `data/<action>/<seq>/`.

### noise — `m.noise(...)` / `vsl2 noise` (+ flag nâng cao)

Sinh bản copy trên đĩa: `data/<action>/<src>` → `data/<action>/<new_seq>/`.
Gaussian noise chỉ trên tọa độ (zero giữ nguyên, visibility giữ nguyên),
`--noise-zeros` phá mask (cân nhắc), `--scale-range` mô phỏng xa/gần,
`--mirror` lật trái↔phải (cả pose, tay, depth, shoulder-offset).
**Cảnh báo leakage**: train đã augment in-memory, đừng trộn data noise vào
tập đánh giá chung với bản gốc.

### classes — `m.classes(...)` / `vsl2 classes`

`list` in bảng class + số sequences; `add` tạo thư mục data; `delete` hỏi
xác nhận (kèm xóa dữ liệu) rồi hỏi xóa model cũ. Sau mọi thay đổi class phải
train lại.

### predict — `m.predict(...)` / `m.predict_features(...)` / `vsl2 predict`

Batch inference trên đúng 12 frames (video → đọc 12 frame đầu; thư mục ảnh →
12 ảnh đầu). **Mới:** nhận trực tiếp 12 vector featurized — list 12 mảng
(356,), ndarray (12, 356), hoặc thư mục seq chứa `0..11.npy` — tự gộp,
chuẩn hóa bằng train stats và cho model predict, bỏ qua MediaPipe hoàn toàn.
Trả `{"raw": từ_dự_đoán_hoặc_rỗng, "probs": [...]}` — rỗng khi không thấy tay
hoặc confidence dưới ngưỡng (`CONFIDENCE_THRESHOLD`, có thể override từng
class qua `CLASS_THRESHOLDS`). Validate shape `(12, 356)`, sai → `ValueError`.

### predict_live — `m.predict_live(...)` / `vsl2 live`

Webcam realtime: sliding window 12 frames → predict → vote buffer
(`SMOOTHING_WINDOW`) + ngưỡng ổn định (`SWITCH_STABLE_FRAMES`) mới commit từ
vào câu. Nhận callback `on_word(word, probs)`.

## Config (`defaults.py`)

Copy toàn bộ hằng số từ `backend/config.py` tại thời điểm tách, đường dẫn
trỏ vào `vsl2/`. Đổi giá trị bằng 3 cách (ưu tiên giảm dần): kwargs của
`m.train(...)` → CLI flags → sửa trực tiếp `defaults.py`. Sau khi sửa
`actions.json` bằng tay, gọi `defaults.reload_actions()`. Biến môi trường
được tôn trọng: `FRAME_STD_SIZE`, `AUGMENT_PRESET` (`base`|`strong`).

Các ngưỡng quan trọng: `SEQUENCE_LEN=12`, `CONFIDENCE_THRESHOLD=0.60`,
`SMOOTHING_WINDOW=3`, `AUGMENT_FACTOR=8`, `EPOCHS=200`, `BATCH_SIZE=16`.

## Định dạng dữ liệu

```
vsl2/data/<action>/<seq>/0.npy ... 11.npy + meta.json
vsl2/weights/model.keras + feature_stats.npz + train_record.json + test_set.npz
```

Mỗi `.npy` là vector float 356-dim (pose 6×(x,y,vis) | tay trái 42 | tay phải
42 | arm_depth 2 | shoulder-offset 252). Frame missing = toàn zero, model bỏ
qua nhờ `Masking(0.0)`. File 104-dim cũ được tự pad lên 356 khi load.

## Quy tắc khi sửa module này

1. **Zero import backend** — tiêu chí pass/fail, chạy trong `vsl2/`:
   `grep -rn "from config\|import config\|vsl_translator\|scripts\._common\|groq\|sys\.path\.insert" --include="*.py" .` phải rỗng.
2. Config mới thêm vào `defaults.py`, không hardcode hằng số trong logic.
3. Chuẩn feature phải khớp weights cũ: `extract()` 356-dim + `feature_stats.npz`
   đi kèm checkpoint trong `vsl2/weights/`.
4. Giữ nguyên thuật toán khi copy từ backend (surgical — không "cải tiến"
   ké); logic mới viết file mới.
