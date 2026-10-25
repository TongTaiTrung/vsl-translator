# Changelog
## Chưa phát hành — Masked attention + cân bằng luồng + velocity (arch v2, 2026-10-25)

### Chẩn đoán (từ train_record baseline cùng ngày)
- train_clean acc 1.0, val acc 0.889, gap +0.11; val confidence mean 0.72, 15/54 mẫu dưới 0.7.
- Val confusion: `xinchao↔diachi` (mỗi chiều 1), `khoer→xinchao` (1), `toi→khochiu` (1), `khochiu→tambiet` (2). Test acc 0.963 nhưng test chỉ 54 mẫu (~1.85%/mẫu), nhiễu cao.
- 3 điểm yếu: (1) `WeightedMean` softmax không trừ mask — frame zero vẫn ăn trọng số attention; (2) shoulder-offset 252/356 dim lấn át hand-shape sau concat; (3) thiếu kênh chuyển động tường minh, GRU tự học velocity từ 12 frame ngắn.

### `backend/src/vsl_translator/model.py`
- `WeightedMean.call(inputs, mask=None)`: cộng `-1e9` vào score của frame bị mask trước softmax (lấy `mask[0]`, cast đúng dtype, broadcast `[..., None]`). Frame miss/rớt không còn kéo lệch vector gộp. Docstring cập nhật mask-aware.
- Layer mới `FrameVelocity` (`@register_keras_serializable`, có `compute_mask` truyền mask qua): `v[t] = x[t] − x[t−1]`, `v[0] = 0`; chỉ tính khi cả hai frame đều non-zero, còn lại zero. Tính trên slice RAW (pose 18 + tay 84) trước LayerNorm để zero pre-norm là zero thật.
- Projection cân bằng luồng: pose → Dense 32, tay 84 → Dense 64, offset 252 → Dense 64, depths giữ 2; concat static 162 + motion (pose_v 18 + hands_v 84 → Dense 48) = 210 vào GRU. Offset không còn đè hand-shape bằng magnitude thô.
- GRU backbone giữ 48/chiều; tổng params trainable 157.069 (tương đương bản cũ 171.313). Verify: build + predict `(2, 8)` + input nửa-mask không lỗi + save/reload `.keras` thành công.

### `backend/config.py`
- `INACTIVE_INJECT_PROB` 0.35 → 0.15: giảm bơm tay nghỉ giả làm các dấu một tay (`toi/khochiu/tambiet`) giống nhau hơn.
- `EPOCHS` 500 → 200: khớp early stop thực tế dừng ~epoch 44 khi LR vẫn ~1e-3, để cosine decay kịp tác dụng.
- `FOCAL_GAMMA` 2.0 → 1.0, `LABEL_SMOOTHING` 0.05 → 0.02: bớt tập trung vào outlier trên data cân bằng (~31 mẫu/class), bớt đè confidence.

### Đồng bộ inference/load
- `VSLInference/inference.py`: vendored `WeightedMean` masked + `FrameVelocity` mới + đăng ký trong `_CUSTOM_OBJECTS`.
- `backend/scripts/profile_robustness.py`, `backend/scripts/stress_test.py`: import + `custom_objects` thêm `vsl>FrameVelocity` (model v2 không load được nếu thiếu).
- Model cũ `model.keras` vẫn load với code mới (không đổi weights) nhưng được train bằng attention cũ → bắt buộc retrain để hưởng lợi.

### Kết quả train arch v2 (record 2026-10-25T17:09, chưa recollect)
- best val_loss 0.321 → 0.245 (epoch 35), val acc 0.889 → 0.907, gap +0.111 → +0.093.
- Val confidence mean 0.72 → 0.89, mẫu dưới 0.7: 15 → 3.
- Cụm `xinchao/diachi` sạch (val 7/7 cả hai); `khochiu` lên 7/7. Lỗi val còn lại dồn vào `tambiet`: `toi→tambiet` (1), `tambiet→toi` (2), `diachi→tambiet` (1).
- Xấu đi: test acc 0.963 → 0.907 (2 → 5 lỗi/54 mẫu, cùng split seed 42); `tambiet` F1 = 0.714 cả val lẫn test, nhầm hai chiều với `toi`/`khochiu`. Nút thắt còn lại là data class này, không phải architecture.
- Việc tiếp theo: recollect `tambiet` +15–20 seq (đa tốc độ/biên độ, 2 người/camera), `toi/khochiu/diachi` mỗi class ~10 seq; rồi đánh giá bằng `kfold_eval.py` thay vì một split 54 mẫu.
## Chưa phát hành — Khung hình chuẩn 720 + upload 720p (2026-10-25)

### Đã làm gì và tại sao
- `backend/config.py`: `FRAME_STD_SIZE` 640 → 720 (square letterbox giữ nguyên, chỉ to hơn). Lý do KHÔNG làm 1280×720 chữ nhật: `aspect_correct()` và `_compute_arm_depth_ratio()` đều nhân `x` với `w/h`; đổi canvas sang chữ nhật (w/h = 1.78 thay vì 1.0) sẽ scale toàn bộ bone vector theo x, lệch phân bố feature canonical mà model + `feature_stats.npz` đã học → bắt buộc retrain và data cũ mất tương thích. Canvas vuông 720 giữ `w/h = 1.0` nên feature không đổi, model cũ chạy nguyên.
- `VSLInference/_config.py`: `FRAME_STD_SIZE` 640 → 720 (giữ đồng bộ training/inference).
- `frontend/static/js/ui/cameraCapture.js`: `MAX_W` 640 → 1280, `getUserMedia` ideal 640×480 → 1280×720. Upload gửi frame 720p lên server; server vẫn letterbox về vuông 720 rồi detect — MediaPipe nhận input nét hơn (scale 720p→720 là 0.5625 thay vì 640p→640), landmark ổn định hơn, khung debug hiển thị to rõ hơn.
- Verify: `normalize_frame` cho mọi nguồn (480p, 720p, 1080p, vuông) đều ra canvas (720, 720, 3); data `.npy` cũ tương thích (pixel đã qua canonical hóa, không phụ thuộc kích thước canvas); test suite giữ nguyên 2 lỗi arm-depth tồn tại từ trước (đã kiểm tra bằng `git stash`).

## Chưa phát hành — Ổn định live (hysteresis + ngưỡng theo class) + preset augment strong (2026-10-25)

### Bối cảnh và mục đích
- Offline bão hòa: train/val/test đều 100% trên 41 mẫu held-out (8 class, ~30 seq/class, tinhthanh 60). Số liệu offline không còn tín hiệu để tune.
- Vấn đề thực tế: chạy live, các từ động tác gần giống nhau bị chập chờn, confidence nhảy, nhãn đổi liên tục. Chưa ghi lại được cặp nhầm cụ thể.
- Ràng buộc đợt này: KHÔNG cloud, KHÔNG thu thêm dữ liệu. Chỉ đụng code model/feature/training/inference.
- Mục tiêu: (1) tái hiện nhầm lẫn offline để đo được, (2) tăng margin phân biệt, (3) ổn định live bằng voting/threshold, (4) mọi thay đổi model báo cáo 3 số: held-out test acc, stress-test acc, per-class min-confidence.

### Chẩn đoán đã làm (quan trọng, đổi hướng cả plan)
- Viết `scripts/stress_test.py`: đánh model hiện tại dưới nhiễu live-like (noise 0.09, time-warp 0.35, frame-drop 0.15, burst mất 1 tay 3–6 frame liên tiếp, speed-cut, partial sliding windows với overlap ≥ 50%).
- Kết quả: model CŨ vẫn 100% trên mọi nhiễu tổng hợp (410 bản stress + 378 partial windows). Nhiễu nhẹ không phải nguyên nhân live kém.
- Viết `scripts/profile_robustness.py`: sweep từng loại nhiễu riêng lẻ (noise 0.05→0.25, warp 0.2→0.7, drop 0.05→0.4, shift, scale-jitter, burst 2→8 frames, speed-cut 1→4, hand_dropout/swap/inject ở p=1.0). Kết quả: TẤT CẢ vẫn 100%.
- Baseline nearest-centroid không train đạt 96% (camon 0.867, khoer 0.800) → bài toán 8 từ hiện tại về bản chất đã dễ, các class cách xa nhau.
- Đo khoảng cách: centroid diachi–tambiet = 9, NHỎ HƠN phân tán nội class (~9–12). Hai từ này chồng lấn nhau trong không gian đặc trưng.
- Confidence val/test trung bình chỉ ~0.80, sát ngưỡng cứng 0.70 → live dễ rớt dưới ngưỡng và nhảy nhãn.
- Kết luận: data 30 mẫu thu trong một phiên/người quá đồng nhất; live khác phiên (tốc độ, biên độ, vị trí đứng, tracking gap) nằm ngoài phân bố train. Hướng fix: augment khắc nghiệt hơn để ép bất biến rộng + hysteresis ở inference.

### Đã khắc phục 1: hysteresis + ngưỡng theo class (chống chập chờn live)
- `backend/config.py`: thêm `SWITCH_STABLE_FRAMES = 2` (challenger phải thắng N vòng vote liên tiếp mới thay dự đoán), `CLASS_THRESHOLDS = {}` (dict rỗng, override từng class khi cần), hàm `threshold_for(idx)` (per-class hoặc fallback 0.70 toàn cục). Chọn `AUGMENT_PRESET` qua env `AUGMENT_PRESET` (mặc định "base", giữ nguyên hành vi cũ).
- `backend/scripts/stream_session.py` (đường live dùng chung cho realtime CLI): thêm state `_challenger/_challenger_runs`; logic vote: winner == pred hiện tại → giữ; winner mới → đếm runs liên tiếp, đủ `SWITCH_STABLE_FRAMES` mới đổi; dưới ngưỡng → pred = -1 và reset challenger; reset challenger khi mất tay ≥ 5 frame.
- `frontend/app/services/predictor.py`: mirror y hệt hysteresis trên (khởi tạo, start/stop inference, nhánh mất tay, nhánh dưới ngưỡng).
- `VSLInference/_config.py` + `VSLInference/inference.py`: parity `threshold_for`, `CLASS_THRESHOLDS`, `SWITCH_STABLE_FRAMES`; single-shot `predict()` dùng `threshold_for(best_idx)` thay ngưỡng cứng.
- Tác dụng: flip-flop A/B từng frame không còn đổi nhãn ngay; chỉ đổi khi cử chỉ mới thắng ổn định liên tiếp → hết chập chờn giữa hai từ giống nhau, giá là trễ ~2 vòng vote (~0.2–0.3s).
- `backend/tests/test_hysteresis.py` (mới, 4 test PASS): giữ dự đoán khi winner ổn định; 1 frame challenger không lật được; challenger thắng đủ `SWITCH_STABLE_FRAMES + SMOOTHING_WINDOW` vòng thì đổi; winner dưới per-class threshold (0.9 < 0.95) thì giữ uncertain (-1).

### Đã khắc phục 2: preset augment "strong" + train lại model final
- `backend/src/vsl_translator/dataset.py`: thêm `_burst_hand_drop()` (zero 1 tay 3–6 frame liên tiếp, mô phỏng MediaPipe mất track giữa chừng), `_speed_cut()` (cắt đầu/cuối 1–3 frame + zero-pad, mô phỏng bấm ký sớm/muộn ở biên cửa sổ); `augment_sequence(preset)` và `augment_train_data(preset)` nhận preset "base"|"strong" (None → `cfg.AUGMENT_PRESET`); base giữ nguyên hệ số cũ, strong dùng `STRONG_*` + xác suất warp 0.7/drop 0.5 + burst 0.6 + speed-cut 0.5. Đo được strong gây nhiễu mạnh gấp ~3.5 lần base (mean |perturb| 0.0728 vs 0.0209).
- `backend/config.py`: hằng số `STRONG_NOISE_STD = 0.09`, `STRONG_SCALE_JITTER_STD = 0.08`, `STRONG_TIME_WARP_SIGMA = 0.35`, `STRONG_SHIFT_RANGE = ±0.04`, `STRONG_FRAME_DROP_PROB = 0.15`, `STRONG_BURST_PROB = 0.6`, `STRONG_BURST_MIN/MAX = 3/6`, `STRONG_SPEED_CUT_PROB = 0.5`, `STRONG_SPEED_CUT_MAX = 3`.
- `backend/scripts/train.py`: ghi `augment_preset` vào `train_record.json` để so sánh thử nghiệm.
- Công cụ đo mới: `scripts/kfold_eval.py` (K-fold chạy đúng pipeline train.py: split stratified → augment train-only → train stats → BiGRU + early stopping; báo mean/worst-fold acc + margin; cờ `--preset`), `scripts/profile_robustness.py` (sweep từng nhiễu riêng lẻ).
- K-fold 3 folds: base 100%/100%/100% (mean 1.0000, margin 0.7714); strong 100%/100%/98.89% (mean 0.9963, margin 0.7858, fold 2 sụt khoer 0.9). Strong làm K-fold SỤT nhẹ nhưng margin CAO hơn — đúng tác dụng: ép model học bất biến rộng hơn thay vì khớp khít phân bố đồng nhất.
- Train final `AUGMENT_PRESET=strong`: held-out test 100% (41/41), val mean conf 0.8078 (min 0.7475), stress + partial windows 100% (378 windows). `models/model.keras` + `feature_stats.npz` + `train_record.json` đã thay (backup cũ xoay vòng tự động trong `models/backups/`; backup `backup_20261022_231039` bị xoay vòng xóa theo `MAX_BACKUPS=5`, chỉ còn stub thư mục).

### Lỗi tồn đọng (chưa sửa, không liên quan đợt này)
- `tests/test_arm_depths.py` còn 2 lỗi TỒN TẠI TỪ TRƯỚC (đã kiểm tra bằng `git stash`: lỗi vẫn hiện khi revert toàn bộ thay đổi đợt này):
  - `ERROR: test_session_calibration` — `ArmDepthCalibrator.update() got an unexpected keyword argument 'shoulder_width_px'` (test gọi API cũ, code đã đổi signature).
  - `FAIL: test_pixel_aspect_invariance` — `0.3 != 1.0` (ratio kẹt ở clip min, kỳ vọng test sai hoặc hằng số đổi mà test chưa theo).
- Full suite: 38 tests (34 cũ + 4 hysteresis mới), chỉ 2 lỗi arm-depth trên; hysteresis 4/4 pass, far_bias + universal_connector pass.
- Chưa làm (hẹn đợt sau, cần train dài hoặc thu data): kênh vận tốc/gia tốc tay, loss contrastive/triplet phụ trên embedding, ablation tắt hand_swap/hand_dropout cho dấu một tay, ngưỡng per-class thực tế cho diachi/tambiet (hiện `CLASS_THRESHOLDS` còn rỗng — cần live test ghi lại cặp nhầm để đặt số).
- Lưu ý live test thủ công vẫn bắt buộc: đếm số lần nhảy nhãn/phút trên các từ giống nhau để xác nhận hysteresis có hết chập chờn ngoài hiện trường.

## Chưa phát hành — Thay face-offset bằng raised-shoulder anchors (188-dim → 356-dim)

### Tại sao dim thay đổi
- Block face-offset cũ (84 số: 42/bên, offset từng điểm tay tới trung tâm mặt)
  bị thay bằng block shoulder-offset (252 số: 126/bên). Mỗi bên tay tính offset
  tới 3 anchors: vai trái nâng + vai phải nâng + trung điểm hai vai, mỗi anchor
  42 số → 126/bên. Layout mới: pose 18 | lh 42 | rh 42 | depths 2 |
  sh_off_lh 126 [104:230] | sh_off_rh 126 [230:356] → `BODY_ONLY_DIM = 356`.
- Lý do bỏ face-center: tâm mặt phụ thuộc tai pose (7, 8) hoặc face mesh
  (33, 263) — khi mặt nghiêng/che/khuất là mất anchor, cả 84 số về 0 dù tay vẫn
  thấy rõ. Vai (11, 12) là anchor bắt buộc đã có của pipeline (mất vai → frame
  0), nên neo theo vai ổn định hơn và không thêm điểm hỏng mới.
- Lý do "nâng lên ngang mặt": giữ x vai (vị trí thân người) nhưng lấy y face
  center → anchor nằm ở độ cao vùng cử chỉ VSL (quanh mặt/ngực), giữ được thông
  tin "tay cao hay thấp so với mặt" mà face-offset cũ nắm, khớp overlay
  `draw_landmarks_test` (nối tay → 2 vai nâng + midpoint, không nối vào mặt nữa).
- Lý do 3 anchors thay vì 1: midpoint vai cho vị trí tổng thể, 2 vai nâng giữ
  thông tin trái/phải (tay đang lệch về vai nào) — cùng palm-normalize
  `(pts − anchor)/palm × 0.4` nên bất biến khoảng cách camera như cũ.

### `landmarks.py`
- Mới `_get_raised_shoulders()`: (ls_raised, rs_raised, mid), giữ x vai + y face
  center; thiếu vai hoặc thiếu mặt → `(None, None, None)`.
- Mới `_shoulder_offsets(hand_lm, anchors, palm_size)`: concat 3 block 42 → 126;
  missing/degenerate → `zeros(126)`.
- `extract()` dùng anchors nâng thay `face_center`; hằng mới `SH_OFF_REF_DIM = 126`,
  `SH_OFF_DIM = 252`, `BODY_ONLY_DIM = 356`. `_face_offsets` giữ lại dạng legacy
  (migration/tests), không còn dùng trong `extract()`.

### `model.py`, `dataset.py`, `config.py`
- `build_model`: nhánh norm `face_n` → `sh_n`, assert 356.
- `dataset.py`: `_random_shift` / `_random_hand_dropout` / `_random_hand_swap` /
  `_side_active` / `inject_inactive_side` chuyển sang `SH_OFF_*`; injection sinh
  3 anchors tổng hợp từ tư thế synthetic thay proxy `(0, -0.8)`; padding 104-dim →
  252 zeros.
- `config.py`: comment layout 356-dim.

### Breaking change (bắt buộc)
- Mọi `data/*/*/*.npy` (188-dim), `models/model.keras`, `models/feature_stats.npz`,
  `models/test_set.npz` vô hiệu → thu lại data, train lại từ đầu. Không trộn
  cũ/mới.

### Test
- `tests/test_far_bias.py`: `TestFaceOffsets` → `TestShoulderOffsets` (khớp công
  thức 3-anchor, thiếu anchor → zeros 252, idle zero block cùng bên);
  `test_arm_depths.py`: assert 356.
- Verify: `venv/bin/python -m unittest tests.test_far_bias tests.test_arm_depths`
  → 26/26 pass; `build_model` cho `input_shape (None, 24, 356)`.
## Chưa phát hành — Sửa kiến trúc model sau khi thêm face-offset (Per-group LayerNorm + attention pooling)

- Per-group LayerNorm đầu vào: slice 188-dim thành 4 nhóm (pose 18 | hands 84
  | depths 2 | face 84) theo hằng `POSE_DIM/HAND_DIM/ARM_DEPTH_DIM`, norm riêng
  từng nhóm rồi concat — face-offset không còn đè hand shape bằng magnitude.
- Attention pooling thay last-state: GRU-2 `return_sequences=True` + Dense-tanh
  → softmax weights → weighted mean qua time — quyết định dùng toàn chuỗi,
  bớt nhảy khi frame cuối rung.
- Giảm width: BiGRU 64→48, Dense 128→64; params ~122k (từ ~200k).
- `Slice`/`WeightedMean` là custom layer có `@register_keras_serializable`,
  giữ mask Masking và save/load `.keras` không cần `custom_objects`.
- Bắt buộc train lại từ đầu (input norm đổi); metric chính: flip-rate nhãn
  trên sliding window + val acc / confusion matrix.
## Chưa phát hành — Bỏ motion features (202-dim → 188-dim)

- Xóa `FeatureStream`, `build_motion_features()`, các hằng `MOTION_*` /
  `FULL_MOTION_DIM`, `FEATURE_VERSION`, `MOTION_VEL_CLIP`, `MOTION_DDEPTH_CLIP`.
- `n_features()` trả 188; train/infer dùng trực tiếp vector `extract()`.
- `dataset.py`: `_coord_mask()` gọn lại, `_random_hand_swap()` bỏ mirror motion,
  `augment_sequence()` và `load_raw_dataset()` bỏ strip/rebuild motion.
- Scripts realtime và `universal_connector.py`: bỏ `FeatureStream`, append
  vector thô vào sliding window.
- Xóa `tests/test_motion_features.py`.
- Lý do: GRU tự học động lực thời gian từ chuỗi thô; đạo hàm viết tay không
  còn cần thiết và là gánh bảo trì (offline/realtime phải giống nhau từng frame).
## Chưa phát hành — Face-offset features: thêm x/y_offset_face_center cho cả 21 điểm mỗi tay (104-dim → 188-dim body, 118 → 202 full)

### Động cơ
- Nhiều ký hiệu VSL diễn ra gần mặt; slot tay local (neo cổ tay) không nói cho
  model biết tay đang ở đâu so với mặt → thêm quan hệ tay–mặt vào feature.

### `landmarks.py`
- Tâm mặt = trung điểm 2 tai pose (landmark 7, 8) qua `_get_joint()` (có sẵn
  visibility gate); fallback trung điểm 2 mắt ngoài face mesh (33, 263);
  cả hai thiếu → block face-offset = 0.
- `_face_offsets(hand_lm, face_center, palm_size)`: `(pts − fc) / palm × 0.4`
  (cùng hệ canonical với slot tay, bất biến khoảng cách camera), 42 số/bên.
- `extract()` concat `[pose 18 | lh 42 | rh 42 | depths 2 | face_lh 42 | face_rh 42]`
  → `BODY_ONLY_DIM = 188`, motion `FULL_MOTION_DIM = 202`. Trailing block nên
  slice cũ `[18:60]/[60:102]/[102:104]` giữ nguyên. Idle-gate zero face-offset
  cùng bên tay idle.
- Hằng mới: `FACE_REF_DIM = 42`, `FACE_OFF_DIM = 84`.

### `dataset.py`, `scripts/add_noise.py`
- `_coord_mask()`: thay literal `42` bằng `HAND_DIM`; face-offset là tọa độ →
  augmentable (noise/scale/shift), vis/ratio exclusion giữ nguyên.
- `_random_shift()` mở rộng is_dx/is_dy cho 2 block face; `_random_hand_dropout()`
  và `_random_hand_swap()` zero/swap face block cùng bên (swap gồm depths đã có).
- `inject_inactive_side()`: tay nghỉ giả cũng sinh face-offset từ cổ tay giả
  (`inject_palm`, `inject_fc`) để phân phối train/infer khớp.
- Compat legacy: `(104,) → (188,)` (lấy `[0:102]` + depths + 84 số 0); lưu ý
  face-offset zero hàng loạt là pattern giả — chỉ dùng để smoke test, không train.

### `universal_connector.py`
- `predict()` raise `ValueError` rõ ràng khi dim trích xuất ≠ dim model, và bỏ qua
  standardization (warn) khi stats cũ lệch dim thay vì crash broadcast.

### Breaking change (bắt buộc)
- Mọi `data/*/*/*.npy` (104-dim), `models/model.keras`, `models/feature_stats.npz`,
  `models/test_set.npz` vô hiệu → thu lại data, train lại từ đầu. Không trộn
  cũ/mới. `scripts/migrate_data.py` đã disable với lý do (face không khôi phục được).

### Test
- `tests/test_far_bias.py`: mock tai pose + face mesh; 3 case mới
  (offsets khớp công thức, thiếu mặt → block 0, idle zero face cùng bên).
- `tests/test_universal_connector.py`: build model tạm theo dim hiện tại thay vì
  nạp artifact cũ trên đĩa.
- Verify: `venv/bin/python -m unittest discover tests` → 45/45 pass.
## Chưa phát hành — Sửa depth tay: đo bằng pixel + calibration theo phiên + hiệu chỉnh z + smoothing realtime

### Nguyên nhân đã xác định
- `_compute_arm_depth_ratio()` so sánh độ dài xương trên tọa độ normalized (phụ thuộc khoảng cách người–camera, cỡ người) với hằng số canonical 2.2 → cùng tư thế nhưng đứng gần/xa cho ratio khác nhau.
- `hypot(dx, dy)` trên tọa độ normalized bị méo theo aspect ratio (x chuẩn hóa theo rộng, y theo cao) → đổi camera/độ phân giải là đổi ratio.
- Không calibration theo từng người/phiên → ngưỡng `< 1.0` không có nghĩa tuyệt đối跨người.
- `extract()` drop hoàn toàn `z` MediaPipe dù đó là tín hiệu depth trực tiếp; depth chỉ suy từ foreshorten 2D, mơ hồ với gập khuỷu trong mặt phẳng ảnh.
- `extract()` pure per-frame, không smoothing → jitter landmark làm tín hiệu push/pull nháy quanh 1.0.
- Idle-gate ép `arm_depths = 1.0` → bước nhảy giả mỗi lần tay đi qua vùng idle.

### `landmarks.py` — đo depth bằng pixel + neutral theo phiên
- `_compute_arm_depth_ratio()` nhận thêm `frame_shape=(h, w)`: nhân `(x*w, y*h)` rồi mới `hypot`; `None` thì giữ đường normalized cũ (tương thích data/test cũ).
- Nhận thêm `neutral_len`: độ dài tay trung tính hiệu chuẩn theo phiên, fallback `2.2` khi không có.
- Nhận thêm `dz = wrist.z − shoulder.z`, hiệu chỉnh phụ `ratio += k*dz` (`k = ARM_DEPTH_Z_GAIN`), clip `[0.3, 2.0]`; z MediaPipe tương đối và nhiễu nên chỉ là hiệu chỉnh phụ.
- `_get_joint()` giữ thêm `z` (trả `(x, y, vis, z)`).
- `ArmDepthCalibrator`: median độ dài tay pixel của `ARM_DEPTH_CAL_WARMUP` frame non-idle (~20) mỗi bên; sanity-clamp ngoài `[1.5, 4.0]` lần shoulder-width thì giữ cũ.
- `ArmDepthSmoother`: EMA (`alpha = ARM_DEPTH_EMA_ALPHA`) + median window (`ARM_DEPTH_SMOOTH_WINDOW`); chỉ dùng ở caller realtime, data train giữ raw.
- `feed_calibrator(results, frame_shape, calibrator)`: helper nạp độ dài tay pixel từng frame.
- `extract(results, frame_shape=None, neutral_lens=None)`: truyền tham số mới xuống hàm ratio; idle-gate chỉ zero pose/hand, depth giữ giá trị đo (bỏ ép 1.0).

### `config.py` — hằng số depth mới
- `ARM_DEPTH_Z_GAIN = 2.0`, `ARM_DEPTH_EMA_ALPHA = 0.4`, `ARM_DEPTH_SMOOTH_WINDOW = 5`, `ARM_DEPTH_CAL_WARMUP = 20`, `ARM_DEPTH_RATIO_MIN/MAX = 0.3/2.0`.

### Scripts — nối calibration + smoothing realtime
- `collect_data.py`, `cam_test.py`: truyền `frame_shape=frame.shape[:2]` vào `extract()`.
- `realtime_predict.py`, `predict_live.py`: mỗi session giữ 1 `ArmDepthCalibrator` + 1 `ArmDepthSmoother`; `feed_calibrator()` trước `extract()`, lọc `vec[102:104]` qua smoother sau `extract()`.

### Tương thích
- Layout 104-dim giữ nguyên (`arm_depths` ở `[102:104]`, semantics ratio quanh 1.0); `_coord_mask` trong `dataset.py` đã exclude 2 kênh này nên không sửa.
- Data train offline cũ dùng đường fallback (normalized + mẫu số 2.2) → không cần migrate; calibration + smoother chỉ kích hoạt ở pipeline mới.

### Test
- Mới trong `test_arm_depths.py`: aspect-invariance 4:3 vs 16:9, calibration cho neutral ≈ 1.0 ở scale khác nhau, dz âm → ratio nhỏ hơn, smoother không nhảy quá 0.15/frame qua spike đơn.
- `test_far_bias.py::TestIdleGate`: depth khi gate không còn bị ép 1.0, giữ giá trị đo.
- Verify: `venv/bin/python -m unittest tests.test_arm_depths tests.test_far_bias` → 23/23 pass, giữ `input_shape (None, 24, 104)`.
## Chưa phát hành — Fix bias nhầm 1-tay thành 2-tay ở xa (translation explosion + idle-arm gate + injection augment)

### Nguyên nhân đã xác định
- `_process_hand()` cộng `wrist_canon` (~2.2 ở xa) vào cả 21 landmark tay → feature tay bùng biên độ, kích hoạt nhầm trọng số 2-tay.
- Neo tay bằng cổ tay pose-detector (`wrist_raw`) thay vì `pts[0]` của hand-detector → sai số detector bị khuếch đại khi chia cho `palm_size` nhỏ.
- Tay buông thõng ở gần thì ngoài khung hình (slot 0), ở xa lọt vào khung hình (slot khác 0) → lệch phân phối train/infer.
- LSTM học shortcut "2 tay khác 0 = hành động 2 tay" vì mẫu 1-tay train luôn zero một bên.

### `landmarks.py` — sửa hand extractor
- `_process_hand(hand_lm)`: bỏ tham số `wrist_raw`/`wrist_canon`; neo tại `w0 = pts[0]` (cổ tay của hand-detector).
- Công thức mới: `normed = (pts - w0) / palm_size * STD_PALM`, không cộng cổ tay canonical (vị trí toàn cục đã nằm ở slot pose).
- Guard palm: `PALM_EPS = 1e-6` → `MIN_PALM_SIZE = 0.012`; dưới ngưỡng trả `zeros(42)`, chống nổ số học.
- `detect()`: bọc `model.process()` trong try/finally, luôn khôi phục `image.flags.writeable = True` kể cả khi exception.
- Cập nhật call site trong `extract()` theo signature mới; layout 102-dim giữ nguyên (pose `[0:18]` | lh `[18:60]` | rh `[60:102]`).

### `landmarks.py` — cổng không gian tay-buông (idle-arm gate)
- Thêm `IDLE_WRIST_Y_MAX = 1.75`, `IDLE_WRIST_X_MAX = 0.95`, đọc qua `getattr(cfg, ...)` với default này.
- Sau FK mỗi bên: nếu `wrist_c[1] > Y_MAX và abs(wrist_c[0]) < X_MAX` → skip bên đó (9 số pose + 42 số tay giữ 0), bên còn lại không ảnh hưởng.
- Tay buông thõng cho feature 0 ổn định dù lọt vào khung hình ở xa.

### `dataset.py` + `config.py` — injection tay-nghỉ lúc train
- Thêm `inject_inactive_side(seq, p=0.35)`: phát hiện mẫu 1-tay (một bên pose+hand toàn 0, bên kia active theo đúng index pose trái `[0:9]` / phải `[9:18]`), inject tay-nghỉ giả lập vào bên trống: cổ tay/khuỷu canonical y 1.1–1.6 (mid-torso), drift Gaussian std 0.015, bàn tay local gần như tĩnh + jitter nhỏ (không inject tọa độ global vào slot tay local), label giữ nguyên.
- Nối vào `augment_sequence()` trên nhánh train-only (`augment_train_data`); `load_raw_dataset` và val/test không chạm tới.
- Config mới: `INACTIVE_INJECT_PROB = 0.35`, `INACTIVE_WRIST_Y_RANGE = (1.1, 1.6)`, `INACTIVE_DRIFT_STD = 0.015`.
- Mục tiêu: ép model học quỹ đạo tay active thay vì shortcut "tay thứ hai khác 0".

### Dataset — không có script convert, phải thu lại
- Không tồn tại script convert landmark/1662-dim → 102-dim (`FULL_DIM` chỉ là comment); `data/*/*/*.npy` đang là feature 102-dim trích bằng logic cũ, không lưu raw landmark → không patch tại chỗ được.
- Thu lại: `scripts/collect_data.py --action <mỗi action>` cho toàn bộ action, rồi train lại. Không trộn feature cũ/mới trong cùng dataset. `data_legacy_*` giữ nguyên archival.

### Breaking change
- `models/model.keras`, `best_model`, `feature_stats.npz`, `test_set.npz` cũ encode phân phối cũ → vô hiệu, bắt buộc retrain. Thu lại toàn bộ `data/` rồi chạy `scripts/train.py`, kiểm tra bằng `scripts/predict.py`.
- Test mới `tests/test_far_bias.py` (repo chưa có test tự động): bao 10 điểm — độc lập tay/wrist, palm guard, writeability, gate đúng bên, detect mẫu 1-tay, p=0.35 + tĩnh-giả, label giữ nguyên, val/test sạch, dim đúng 102, tương thích checkpoint.
## Chưa phát hành — Canonical Skeleton Retargeting (102-dim)
- `landmarks.py` — `extract()` viết lại bằng Forward Kinematics: vai canonical
  `(∓0.5, 0)`, khuỷu = vai + unit(xương cánh tay)×1.2, cổ tay = khuỷu +
  unit(xương cẳng tay)×1.0; tay chuẩn hóa bằng `palm_size` riêng
  (`max(|w0−mcp9|, |mcp5−mcp17|×1.25)`) và neo tại cổ tay canonical.
  Output 120 → **102-dim**: pose `[0:18]` | lh `[18:60]` | rh `[60:102]`.
- Fallback per-side: mất vai → zero cả frame; mất khuỷu/cổ tay bên nào →
  zero bên đó (2 khớp pose + 42 số tay), bên còn lại giữ nguyên.
- `data/` 120-dim archive → `data_legacy_120/`; model/stats/test cũ đã xóa —
  thu lại toàn bộ, train từ đầu. Augment (`dataset.py`, `add_noise.py`) không
  đổi logic (dùng `POSE_DIM`/`HAND_DIM` symbolic, pose giữ stride-3).
- Verify bằng mock: bất biến khoảng cách chính xác, tay đứng yên khác 0,
  low-vis/mất landmark zero đúng, mask augment đúng kênh vis mới.

## Chưa phát hành — Train pipeline: chống rò rỉ add_noise + metric sạch
- Rò rỉ nghiêm trọng đã xử lý: các copy do `add_noise.py` sinh (seq 6–30 mọi action) được dời sang `data_noise_backup/` — `data/` chỉ còn 5 seq thu tay/action; đa dạng hóa chỉ sinh in-memory qua `augment_train_data()` sau split. Xóa `models/model.keras`, `feature_stats.npz`, `test_set.npz` cũ (nhiễm leakage) — cần train lại.
- Docstring `train.py` 80/20 → mô tả đúng 3-way split (test trước, rồi val trên phần còn lại).
- Thêm metric train-clean (raw train chuẩn hóa bằng train stats) bên cạnh train-aug; gap clean-val phản ánh khái quát hóa thật.
- Val report hạ xuống tuning signal (lạc quan do chọn checkpoint theo val); held-out test acc là kết quả chính.
- Class weight giữ tính sau augment (đúng phân phối optimizer thấy) + comment giải thích.
- Xóa `steps_per_epoch` chỉ in log không dùng; `overfit_patience=30` hardcode → `cfg.OVERFIT_PATIENCE`.
- Quy tắc từ nay: không chạy `add_noise.py` vào `data/` trước khi train; muốn đa dạng khoảng cách thì thu thật 2–3 khoảng cách hoặc dùng augment in-memory.

## Chưa phát hành — `add_noise.py`: nhiễu khoảng cách + noise giá trị 0
- Thêm `apply_distance_scale(vec, factor)`: nhân toàn bộ offset (dx,dy) với 1 hệ số, visibility giữ nguyên, 0 vẫn 0.
- Cờ `--scale-range LOW HIGH`: mỗi copy rút 1 factor `uniform(LOW, HIGH)` rồi scale toàn sequence trước khi noise (thứ tự scale → noise). Khuyến nghị `0.7 1.4`; range rộng (< 0.5 hoặc > 2.0) in warning.
- Cờ `--noise-zeros` (mặc định off): noise cả entry 0 trên kênh tọa độ, vẫn chừa visibility; in warning vì phá ngữ nghĩa mask 0.0 = missing (Masking không bỏ qua được nữa) — dùng liều thấp, so confusion matrix.
- Tái cấu trúc `_coord_mask()` dùng chung cho `add_noise()` + `apply_distance_scale()`. Khác `_scale_jitter()` trong `dataset.py` (per-frame Gaussian ~N(1,0.05)) — script là per-sequence uniform, không gộp.
- Verify: scale x2 đúng tỉ lệ trên kênh tọa độ, visibility + 0 giữ nguyên; `--noise-zeros` chạm 0 nhưng chừa vis; chạy thật `--src-seqs 1 --copies 1 --scale-range 0.7 1.4` sinh `hello/11 [x0.96]` đủ 24 frame (đã xóa mẫu test).
- Cờ `--all-actions`: xử lý mọi action trong `cfg.ACTIONS` trong một lệnh (id mới nối tiếp riêng từng action, `--src-seqs` bị bỏ qua ở mode này). Ví dụ: `scripts/add_noise.py --all-actions --copies 2 --scale-range 0.7 1.4 --seed 42`. Logic từng action tách thành `_process_action()` trả về số copy đã tạo.
- Cảnh báo trung thực: scale đồng nhất không mô phỏng được phối cảnh thật hay tracking kém ở xa — vẫn cần collect thật 2–3 khoảng cách để kiểm chứng.

## Chưa phát hành — Phá hand-presence shortcut (code-only, không thu data mới)
- Nguyên nhân: hello collect tay-phải-only, nhanvien tay-trái-only — đo được `lh_zero=1.00` (hello) và `rh_zero=1.00` (nhanvien) trên cả `data/` (120-dim, 40 seq/class) lẫn `data_legacy_183/`. Khối tay mất là 42 số 0 tuyệt đối, giữ nguyên qua standardize + mọi augment; `Masking(0.0)` chỉ mask cả timestep nên frame một-tay đi qua LSTM — model học "khối nào khác 0" thay vì động tác. Realtime 2 tay là mẫu ngoài phân phối → bias về nhanvien + flicker.
- `config.py`: `HAND_DROPOUT_PROB` 0.15 → 0.5 (mỗi mẫu thấy cả bản đủ tay lẫn mất tay); thêm `HAND_SWAP_PROB = 0.3`.
- `dataset.py`: thêm `_random_hand_swap()` — swap 2 khối tay lh↔rh theo `POSE_DIM`/`HAND_DIM` symbolic, giữ label, khối 0 swap vẫn 0; nối vào `augment_sequence()` làm bước 7 sau hand-dropout.
- Inference giữ nguyên (nhận cả 2 tay). Lưu ý trung thực: augment không sinh được mẫu 2-tay cùng cử động từ gốc 1-tay, chỉ phá shortcut phía-tay; realtime 2-tay vẫn là ngoại suy nhưng gần phân phối train hơn.
- Verify: 1000 bản augment từ mẫu mock 1-tay-phải → gốc RH 53%, swap→LH 22%, dropout→0-tay 25%; swap 2 lần = identity; shape (24,120).
- `models/model.keras`, `feature_stats.npz`, `test_set.npz` đã xóa — train lại từ `data/` hiện tại. Rủi ro: class sau này phân biệt bằng tay trái/phải thật thì phải tắt swap.

## Chưa phát hành — Refactor 120-dim: pose vai/tay, scale rộng-vai thuần
- `landmarks.py` — xóa `_get_hips()`, `torso_length()`, `reference_scale()`, hằng `L_HIP`/`R_HIP`; scale chuẩn hóa chỉ dùng `shoulder_width()` + guard `MIN_SHOULDER_WIDTH` (dưới ngưỡng hoặc mất vai → frame 0, không fallback).
- `extract()` chỉ lấy pose MediaPipe indices 11–22 (`lm[POSE_START:POSE_END]`, 12 điểm vai/khuỷu/cổ tay/bàn tay pose); bỏ face (0–10) và thân/dưới (23–32).
- Vector 183 → **120-dim**: pose 12×(dx,dy,vis)=36 `[0:36]` | lh 42 `[36:78]` | rh 42 `[78:120]`. `POSE_DIM = 12 * 3`, `BODY_ONLY_DIM = 120`. Profile view (vai trùng X) → frame 0 có chủ đích, thu action frontal.
- `dataset.py` `_time_warp()`: kênh visibility không nội suy tuyến tính nữa — lấy nearest-frame rồi re-zero theo mask, chống interpolation bleed. Đã sửa lỗi boolean-indexing khi re-zero.
- `config.py`: xóa `MIN_TORSO_LENGTH`; cập nhật comment layout 120-dim. `__init__.py`: bỏ export `torso_length`, `reference_scale`.
- Breaking change: `data/` 183-dim (1440 file hello/nhanvien) archive sang `data_legacy_183/`; `data/` trống; `models/model.keras`, `feature_stats.npz`, `test_set.npz` đã xóa — thu lại toàn bộ rồi train từ đầu.
- Verify: frontal → 120-dim, wrist dx=1.000; mất vai → 0; profile → 0; `_coord_mask(120)` vis đúng idx 2,5,...,35; noise/shift/scale-jitter giữ visibility; warp không bleed giá trị mới.

## Chưa phát hành — Củng cố bất biến khoảng cách (audit plan)

### Nguyên nhân đã xác định (audit)
- Vai dùng vô điều kiện, không kiểm tra visibility: vai bị che/kém tin cậy vẫn tạo `mid`/`scale` rác, làm hỏng cả frame 183-dim.
- Guard `scale < 1e-6` quá lỏng: ở xa `shoulder_width` chỉ ~0.03–0.08, phép chia khuếch đại jitter 12–30 lần mà vẫn lọt qua guard.
- Augment sai đơn vị: `_add_noise` cộng nhiễu cả vào kênh visibility (xác suất [0,1], không phải tọa độ); `_random_shift` cộng 1 scalar vào mọi entry gồm visibility, dx/dy cùng delta nên chỉ dịch chéo.
- Tham số augment đặt theo đơn vị ảnh cũ [0,1] trong khi offset sau chuẩn hóa rộng-vai có biên ~±2–3, khiến augment gần như vô hiệu.
- `shoulder_width` dùng hypot 2D trên tọa độ dị hướng, phụ thuộc aspect ratio và tư thế nghiêng.
- Tàn dư: gọi `select_features()` no-op, `CONFIDENCE_THRESHOLD` local đè `cfg`, `predict_live.py` vẫn vẽ face mesh cũ.

### `landmarks.py` — làm vững mẫu số và gốc tọa độ
- Thêm `_get_shoulders()`: trả về 2 vai chỉ khi cả hai tồn tại và `visibility >= SHOULDER_VIS_THRESHOLD` (0.5); ngược lại trả `(None, None)`.
- `shoulder_midpoint()` / `shoulder_width()` dùng chung `_get_shoulders()`; vai yếu hoặc mất → `extract()` trả frame toàn 0 để lớp `Masking` bỏ qua.
- `shoulder_width()` chuyển từ hypot 2D sang `|x_ls - x_rs|`: ổn định khi nghiêng người, không phụ thuộc aspect ratio khung hình.
- Guard scale `1e-6` → `MIN_SHOULDER_WIDTH = 0.05` (đơn vị ảnh): dưới ngưỡng → frame 0, tránh khuếch đại jitter khi đứng xa.
- Scale lai chống mù khi xoay người (profile view làm `|dx|` vai tiến về 0): thêm `_get_hips()` (gate visibility hông cùng ngưỡng vai), `torso_length()` = `|mid_hip_y - mid_shoulder_y|` thuần trục Y, và `reference_scale()` trả `(scale, source)` — `shoulder` khi `|dx|` vai ≥ 0.05, fallback `torso` khi `|dy|` thân ≥ `MIN_TORSO_LENGTH = 0.08`, cả hai suy biến → frame 0. `extract()` chia offset cho scale lai thay vì rộng vai thuần. Overlay spokes đổi text `shoulder_w` → `scale[source]`. Không nâng `MIN_SHOULDER_WIDTH` lên 0.1–0.15 vì frame xa thật chỉ ~0.03–0.08, nâng ngưỡng sẽ làm mù đúng tập xa.

### `dataset.py` + `config.py` — augment đúng đơn vị rộng-vai
- Thêm `_coord_mask()`: mặt nạ True cho kênh (dx,dy), False cho kênh visibility pose (mỗi kênh thứ 3 trong 99 kênh pose).
- `_add_noise()`: chỉ cộng nhiễu Gaussian vào entry vừa khác 0 vừa là kênh tọa độ; visibility giữ nguyên.
- `_random_shift()`: viết lại thành shift 2D riêng `(sx, sy)` rút 1 lần/sequence, cộng vào kênh dx/dy tương ứng, bỏ qua entry 0 và kênh visibility; mô phỏng jitter của trung điểm vai.
- Thêm `_scale_jitter()`: nhân toàn bộ offset mỗi frame với hệ số ~N(1, 0.05), mô phỏng jitter đo rộng vai; frame toàn 0 nhân lên vẫn 0.
- `augment_sequence()`: thứ tự noise → scale-jitter (50%) → time-warp (50%) → shift 2D (50%) → frame-drop (30%) → hand-dropout.
- Helper `_add_noise` / `_scale_jitter` / `_random_shift` chạy đúng cả sequence 2D `(24, 183)` lẫn batch 3D.
- Hiệu chuẩn theo đơn vị rộng-vai: `NOISE_STD` 0.02 → 0.05, `SHIFT_RANGE` ±0.05 → ±0.02; thêm `SHOULDER_VIS_THRESHOLD`, `MIN_SHOULDER_WIDTH`, `SCALE_JITTER_STD`.
- Xóa gọi `select_features()` chết trong `load_raw_dataset()`.

### Scripts
- `scripts/add_noise.py`: hàm `add_noise()` visibility-safe — dựng mặt nạ tọa độ từ `POSE_DIM`, chỉ noise entry khác 0 trên kênh (dx,dy).
- `scripts/collect_data.py`: đếm `zero_frames` mỗi sequence và in `Seq N: k/24 zero frames (missing shoulders)` để biết tỷ lệ frame chết khi thu.
- `scripts/realtime_predict.py`: gộp majority-vote (`SMOOTHING_WINDOW`) với hand-gating sẵn có — vote trên `(label, conf)` các window gần nhất, label thắng theo số phiếu, confidence là trung bình các phiếu của label thắng; dùng `cfg.CONFIDENCE_THRESHOLD` thay hằng local 0.70; `votes.clear()` cùng buffer khi mất tay ≥ 5 frame; bỏ `select_features` chết.
- `scripts/predict_live.py`: chuyển `draw_landmarks` → `draw_landmarks_new` cho khớp feature 183-dim (pose + 2 tay + spokes, không face).

### Kiểm chứng đã chạy
- Cùng gesture giả lập gần (w=0.2) vs xa (w=0.1): `tip_dx=1.000`, `tip_dy=0.000` cả hai, dim 183 — bất biến scale giữ nguyên.
- `augment_sequence` trên sequence 2D ngẫu nhiên: kênh visibility giữ nguyên tuyệt đối (`vis_untouched: True`), shape `(24, 183)`.
- Vai visibility 0.1 (< 0.5): `extract()` trả frame toàn 0 (`lowvis_zero: True`).
- Scale lai: frontal (vai 0.2, torso 0.3) → `shoulder` tip_dx=1.000; profile (vai 0.01, torso 0.3) → `torso` tip_dx=0.667 (không mù); cả hai suy biến → frame 0; xa frontal (vai 0.06) vẫn đi nhánh `shoulder`.
- Config in ra: `SHOULDER_VIS_THRESHOLD=0.5`, `MIN_SHOULDER_WIDTH=0.05`, `NOISE_STD=0.05`, `SCALE_JITTER_STD=0.05`, `SHIFT_RANGE=(-0.02, 0.02)`.

### Việc còn lại (làm thủ công)
- Thu data mỗi class ở 2–3 khoảng cách (dùng overlay `shoulder_w` để kiểm soát dải giá trị), chú ý log zero-frame.
- Xóa model/stats/test cũ, train lại từ đầu.
- Test riêng tập xa (confusion matrix + mean confidence); realtime lùi xa dần kiểm tra label ổn định, không flicker class.

## Chưa phát hành — Scale normalize bất biến khoảng cách camera
- `landmarks.py` — `extract()` chia mọi offset cho rộng vai (`shoulder_width()` mới):
  `(dx,dy) = (landmark - trung điểm vai) / khoảng cách 2 vai`. Guard `scale < 1e-6` hoặc thiếu vai → frame toàn 0.
  Kiểm tra: cùng gesture ở vai cách 0.2 (gần) vs 0.1 (xa) → vector bằng nhau (`invariant: True`).
- `dataset.py` — xóa `_random_scale()` và lời gọi trong `augment_sequence()` (chuẩn hóa đã triệt tiêu scale, scale giả gây nhiễu); `config.py` xóa `SCALE_RANGE`.
- `draw_shoulder_spokes()` overlay thêm text `shoulder_w` để thấy khoảng cách vai lúc collect.
- Data offset thô (`data/` cũ, max ~1.93) archive sang `data_legacy_unscaled/`; `data/` trống, model/stats/test_set cũ đã xóa — thu lại toàn bộ rồi train từ đầu.
- Thêm `draw_landmarks_new(image, results)` (`landmarks.py`): vẽ pose + 2 tay + spokes về trung điểm vai, không vẽ face — khớp đúng 183-dim `extract()` dùng. `collect_data.py` chuyển sang dùng hàm này.

## 2026-10-19 — Chuyển pipeline sang đặc trưng tương đối vai (183-dim)

### Thay đổi lớn (breaking change)
- `src/vsl_translator/landmarks.py` — `extract()` viết lại hoàn toàn:
  - Bỏ trục z, bỏ face landmark, `(x, y)` thành offset `(dx, dy)` so với trung điểm 2 vai (pose landmark 11/12).
  - Giữ `visibility` của pose. Format mới: pose 33×(dx,dy,vis)=99 + 2 tay 21×(dx,dy)=42+42 → **183-dim** (trước: 1662-dim tuyệt đối, 258-dim sau select).
  - Thiếu 1 trong 2 vai → frame toàn 0 để `Masking` bỏ qua.
  - `select_features()` thành no-op (giữ hàm để import cũ không vỡ); `n_features()` trả 183.
- `data/` cũ (tọa độ tuyệt đối) archive sang `data_legacy_abs/`; `data/` trống, cần thu lại.
- `models/model.keras`, `feature_stats.npz`, `test_set.npz` cũ vô hiệu (sai dim) — bắt buộc retrain từ đầu.
- `scripts/predict.py` — bỏ shim convert 1662→258, sai dim thì báo lỗi retrain.

### Model & training (`model.py`, `config.py`)
- `BatchNormalization` → `LayerNormalization` (4 lớp): BN nhiễm frame mask 0 và moving-average nhiễu trên dataset nhỏ.
- `FOCAL_ALPHA`: 0.25 → 1.0 (alpha scalar không cân bằng class, chỉ thu nhỏ loss; class đã cân bằng 30 mẫu/class + đã có `class_weight`).
- `LABEL_SMOOTHING`: 0.1 → 0.05 (0.1 đè confidence dưới ngưỡng inference 0.70).
- Bỏ `getattr(cfg, ..., default)`, gọi trực tiếp `cfg.X` để lộ lỗi typo config.
- Xóa `USE_FACE` khỏi `config.py` (face không còn trong pipeline).
- Early stopping nới lỏng cho dataset nhỏ (~90 mẫu, val ~13 mẫu rất nhiễu):
  - `ES_START_EPOCH` 15 → 30, `ES_PLATEAU_PATIENCE` 15 → 30,
    `ES_LOSS_SPIKE_PATIENCE` 8 → 15, `ES_PATIENCE` 50 → 60.
- `AUGMENT_FACTOR` 5 → 8 (train raw ~60 mẫu → ~500 mẫu augment).

### Dataset (`dataset.py`)
- `_random_hand_dropout()` theo layout mới pose [0:99] | lh [99:141] | rh [141:183]; bỏ nhánh legacy 1662-dim.

### Inference
- `scripts/realtime_predict.py` — thêm `select_features()` còn thiếu (trước đây đưa 1662-dim vào model 258-dim); nguyên nhân trực tiếp gây nhận diện sai/chập chờn.

### Script mới
- `scripts/add_noise.py` — sinh bản copy nhiễu Gaussian từ sequence có sẵn:
  `venv/bin/python scripts/add_noise.py --action hello --copies 1 --noise-std 0.02`
  (tham số: `--src-seqs`, `--seed`; nhiễu chỉ cộng vào keypoint ≠ 0, giữ mask).
- `scripts/realtime_test.py` — viết lại: in liên tục khoảng cách + offset `dx`/`dy` đầu ngón trỏ tay phải so với trung điểm vai; vẽ line nối 2 điểm, nhãn tọa độ và overlay `dist/dx/dy/tip/mid` lên màn hình.

### Việc còn lại
1. Thu lại data: `collect_data.py` cho mỗi action (tự ra format 183-dim).
2. Xóa model/stats cũ, chạy `train.py` từ đầu.
3. Kiểm tra: confusion matrix + held-out test acc; đứng lệch khung hình khi realtime — label phải ổn định (bất biến vị trí).

Bạn là một lập trình viên Front-end bậc thầy và một nghệ sĩ UI/UX thiên tài. Hãy tạo ra một trang web HTML single-file hoàn chỉnh, tuyệt đẹp, mang phong cách "dreamy", huyền ảo và lãng mạn để chúc mừng ngày Phụ nữ Việt Nam 20/10 cho các bạn nữ trong lớp.

Toàn bộ mã nguồn (HTML, CSS, JavaScript) phải được gom gọn trong **một file duy nhất**, sử dụng CDN cho các thư viện (như Three.js cho hiệu ứng 3D, Tailwind CSS hoặc CSS thuần tùy chỉnh, Canvas cho hiệu ứng nền).

### 1. Yêu cầu về Giao diện & Không gian (Visual & Atmosphere):
- **Nền (Background):** Màu tối sâu thẳm (deep dark/midnight purple hoặc galaxy gradient) kết hợp với hiệu ứng các hạt bụi sáng lấp lánh (starfield/magic dust) bay lơ lửng nhẹ nhàng phía sau để tạo chiều sâu và cảm giác lung linh.
- **Hiệu ứng chung:** Glassmorphism tinh tế cho các thành phần UI, hiệu ứng glow rực rỡ, mộng mơ (soft pink, peach, magenta, gold accents), không gian trang nhã, không bị sến súa quá đà nhưng cực kỳ tinh tế và cảm xúc.

### 2. Tâm điểm - Bông hoa hồng 3D (3D Glowing Rose):
- Ở chính giữa màn hình phải có một **bông hoa hồng 3D** được dựng bằng Three.js (hoặc Shader/Canvas tùy biến cao cấp).
- **Chuyển động:** Hoa hồng tự động xoay nhẹ nhàng 360 độ một cách uyển chuyển, thanh thoát.
- **Hiệu ứng sáng:** Hoa hồng phải có hiệu ứng **glowing rực rỡ** (phát sáng hào quang mềm mại xung quanh, bloom effect), các cánh hoa có độ bóng bẩy, màu hồng pastel/đỏ ruby phối ánh kim tuyến hoặc ánh sáng huyền ảo.
- **Tương tác nhẹ:** Khi rê chuột hoặc chạm vào hoa hồng, nó có thể phản hồi tốc độ xoay hoặc lóe sáng nhẹ.


