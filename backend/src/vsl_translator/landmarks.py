"""
MediaPipe Holistic helpers: detection, landmark drawing, and keypoint extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
import math

mp_holistic = mp.solutions.holistic        # holistic model
mp_drawing = mp.solutions.drawing_utils    # drawing utilities


def create_model():
    """Khởi tạo Holistic với cấu hình tối ưu từ config (complexity 2, smooth, ngưỡng thấp)."""
    import config as cfg
    return mp_holistic.Holistic(
        model_complexity=getattr(cfg, 'MODEL_COMPLEXITY', 2),
        smooth_landmarks=getattr(cfg, 'SMOOTH_LANDMARKS', True),
        min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
    )


def detect(image, model):
    """Run MediaPipe Holistic inference on a BGR frame.

    Pipeline: BGR → RGB → not-writeable → inference → writeable → RGB → BGR
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    try:
        results = model.process(rgb)
    finally:
        rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results


def normalize_frame(frame, size=None):
    """Letterbox any frame to a square standard canvas, preserving aspect.

    Returns (frame_std, meta) with meta = {src_w, src_h, size, scale, pad_x,
    pad_y}. Same input pixels always map to the same output regardless of
    source resolution. The output is what detect()/extract() callers should
    pass as frame / frame_shape.
    """
    import config as _cfg
    if size is None:
        size = int(getattr(_cfg, 'FRAME_STD_SIZE', 640))
    h, w = frame.shape[:2]
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=frame.dtype)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, {"src_w": w, "src_h": h, "size": size,
                    "scale": scale, "pad_x": pad_x, "pad_y": pad_y}


def aspect_correct(x, y, frame_shape):
    """Map normalized (x, y) to aspect-corrected centered coords.

    x' = (x - 0.5) * (w / h), y' = y - 0.5. On letterboxed square frames
    (w == h) this reduces to a plain centering shift, so bone vectors and
    distances are comparable across cameras with different resolutions and
    aspect ratios.
    """
    h, w = frame_shape
    return ((float(x) - 0.5) * (w / h), float(y) - 0.5)


def draw_landmarks(image, results):
    """Draw face, pose, and hand landmarks on the image."""
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    return image


def draw_landmarks_new(image, results):
    """Draw pose + both hands, shoulder spokes, and raised-shoulder spokes.

    Matches the canonical feature set used by extract(): retargeted arm pose
    joints only, no face, no torso, no z-axis. Shoulder spokes visualize the
    shoulder anchor; raised-shoulder spokes visualize the raw hand→anchor
    vectors behind the shoulder-offset feature blocks (two raised shoulders
    + their midpoint).
    """
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    # Spokes from each hand landmark to the shoulder midpoint
    draw_shoulder_spokes(image, results)
    # Spokes from each hand landmark to the raised-shoulder anchors
    # (shoulder-offset features)
    draw_landmarks_test(image, results)
    return image

L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
# Pose ears (face-center anchor, same normalized coords as hands).
L_EAR = 7
R_EAR = 8
# Face-mesh fallback for the face center (outer eye corners).
L_EYE_OUTER = 33
R_EYE_OUTER = 263
# Canonical skeleton: shoulders fixed at (∓STD_HALF_SHOULDER, 0); arm joints
# rebuilt by Forward Kinematics from unit bone vectors, so output is invariant
# to body proportions and camera distance while preserving posture direction.
STD_HALF_SHOULDER = 0.5
STD_UPPER_ARM = 1.2
STD_FOREARM = 1.0
STD_PALM = 0.4
BONE_EPS = 1e-6   # zero-length bone guard for _unit_vec
MIN_PALM_SIZE = 0.012  # degenerate-palm guard for _process_hand (aspect-corrected units)
# Idle-arm spatial gate defaults (overridable via config.py, read with getattr).
IDLE_WRIST_Y_MAX = 1.75
IDLE_WRIST_X_MAX = 0.95


def _get_shoulders(results):
    """Return (ls, rs) if both shoulders exist and are trusted, else (None, None)."""
    import config as cfg
    if results.pose_landmarks is None:
        return None, None
    try:
        ls = results.pose_landmarks.landmark[L_SHOULDER]
        rs = results.pose_landmarks.landmark[R_SHOULDER]
    except IndexError:
        return None, None
    if ls is None or rs is None:
        return None, None
    vis_thresh = getattr(cfg, 'SHOULDER_VIS_THRESHOLD', 0.5)
    if (getattr(ls, 'visibility', 1.0) < vis_thresh
            or getattr(rs, 'visibility', 1.0) < vis_thresh):
        return None, None
    return ls, rs


def shoulder_midpoint(results):
    """Midpoint of the two shoulders in normalized (x, y), or None if missing."""
    ls, rs = _get_shoulders(results)
    if ls is None or rs is None:
        return None
    return ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)


def shoulder_width(results):
    """Horizontal distance between shoulders in normalized coords, or None.

    Uses |x_ls - x_rs| only (not 2D hypot): stable under body tilt and
    independent of frame aspect ratio.
    """
    ls, rs = _get_shoulders(results)
    if ls is None or rs is None:
        return None
    return float(abs(ls.x - rs.x))


def draw_face_spokes(image, results, color=(255, 0, 255), thickness=1):
    """Draw lines from each hand landmark to the face center.

    Debug overlay visualizing the face-offset features: each spoke is the
    raw-space vector that _face_offsets() normalizes by palm_size. Drawn per
    hand so left/right spokes can be told apart by which hand they start from.
    Missing face center → nothing drawn.
    """
    import config as _cfg
    vis_thresh = getattr(_cfg, 'SHOULDER_VIS_THRESHOLD', 0.5)
    fc = _get_face_center(results, vis_thresh)
    if fc is None:
        return image
    h, w = image.shape[:2]
    p_fc = (int(fc[0] * w), int(fc[1] * h))
    for hand_lm in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_lm is None:
            continue
        for res in hand_lm.landmark:
            cv2.line(image, (int(res.x * w), int(res.y * h)), p_fc, color, thickness)
    cv2.circle(image, p_fc, 5, color, -1)
    return image


def draw_shoulder_spokes(image, results, color=(0, 255, 255), thickness=1):
    """Draw lines from each hand landmark to the shoulder midpoint.

    Debug overlay showing the shoulder anchor used by the pipeline.
    Note: extract() stores canonical-FK pose + palm-normalized hands anchored
    at the canonical wrist, not raw offsets to this midpoint.
    """
    mid = shoulder_midpoint(results)
    if mid is None:
        return image
    h, w = image.shape[:2]
    p_mid = (int(mid[0] * w), int(mid[1] * h))
    for hand_lm in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_lm is None:
            continue
        for res in hand_lm.landmark:
            cv2.line(image, (int(res.x * w), int(res.y * h)), p_mid, color, thickness)
    cv2.circle(image, p_mid, 5, (255, 0, 0), -1)
    scale = shoulder_width(results)
    if scale is not None:
        cv2.putText(image, f"shoulder_w: {scale:.3f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def draw_landmarks_test(image, results, color=(0, 255, 0), thickness=1):
    """Test overlay: hand landmarks -> raised shoulder points + shoulder midpoint.

    - Lấy hai vai (pose 11, 12), giữ nguyên x, kéo y lên ngang với mặt
      (y = face center y từ _get_face_center).
    - Nối mỗi landmark tay tới cả hai điểm vai đã nâng + tới trung điểm
      hai vai. Không nối tới giữa mặt nữa.
    - Vai/face thiếu → không vẽ gì.
    """
    import config as _cfg
    vis_thresh = getattr(_cfg, 'SHOULDER_VIS_THRESHOLD', 0.5)
    ls, rs = _get_shoulders(results)
    if ls is None or rs is None:
        return image
    fc = _get_face_center(results, vis_thresh)
    if fc is None:
        return image
    h, w = image.shape[:2]
    # Hai điểm vai kéo lên ngang mặt: giữ x vai, y = y mặt.
    p_ls_raised = (int(float(ls.x) * w), int(float(fc[1]) * h))
    p_rs_raised = (int(float(rs.x) * w), int(float(fc[1]) * h))
    mid = shoulder_midpoint(results)
    p_mid = (int(mid[0] * w), int(mid[1] * h)) if mid is not None else None
    for hand_lm in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_lm is None:
            continue
        for res in hand_lm.landmark:
            p_hand = (int(res.x * w), int(res.y * h))
            cv2.line(image, p_hand, p_ls_raised, color, thickness)
            cv2.line(image, p_hand, p_rs_raised, color, thickness)
            if p_mid is not None:
                cv2.line(image, p_hand, p_mid, color, thickness)
    cv2.circle(image, p_ls_raised, 5, color, -1)
    cv2.circle(image, p_rs_raised, 5, color, -1)
    if p_mid is not None:
        cv2.circle(image, p_mid, 5, (255, 0, 0), -1)
    return image


def draw_arm_lengths(image, results, color=(0, 255, 0), thickness=2,
                     font_scale=0.5, font_color=(255, 255, 255)):
    """Draw arm bone lengths (shoulder→elbow, elbow→wrist) on the image.

    For each visible arm, draws the Euclidean distance (in normalized coords)
    along the bone segment and labels it on the image.

    Args:
        image: BGR frame (modified in-place).
        results: MediaPipe Holistic results.
        color: BGR color for the bone line overlay.
        thickness: Line thickness for the bone overlay.
        font_scale: Font scale for the distance text.
        font_color: BGR color for the distance text.

    Returns:
        The annotated image.
    """
    import config as _cfg
    vis_thresh = getattr(_cfg, 'SHOULDER_VIS_THRESHOLD', 0.5)

    if results.pose_landmarks is None:
        return image

    lm = results.pose_landmarks.landmark
    h, w = image.shape[:2]

    # Define arm segments: (start_idx, end_idx, label)
    arm_segments = [
        (L_SHOULDER, L_ELBOW, "L Upper Arm"),
        (L_ELBOW,    L_WRIST, "L Forearm"),
        (R_SHOULDER, R_ELBOW, "R Upper Arm"),
        (R_ELBOW,    R_WRIST, "R Forearm"),
    ]

    y_offset = 50  # starting y position for the text summary panel
    for start_idx, end_idx, label in arm_segments:
        start_j = _get_joint(lm, start_idx, vis_thresh)
        end_j   = _get_joint(lm, end_idx,   vis_thresh)
        if start_j is None or end_j is None:
            continue

        # Euclidean distance in normalized coordinates
        dx = end_j[0] - start_j[0]
        dy = end_j[1] - start_j[1]
        dist = float(np.hypot(dx, dy))

        # Pixel coordinates
        p1 = (int(start_j[0] * w), int(start_j[1] * h))
        p2 = (int(end_j[0] * w),   int(end_j[1] * h))

        # Draw highlighted bone line
        cv2.line(image, p1, p2, color, thickness)

        # Put distance text at the midpoint of the bone
        mid_px = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(image, f"{dist:.3f}", mid_px,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1,
                    cv2.LINE_AA)

        # Also render a summary panel in the top-left corner
        cv2.putText(image, f"{label}: {dist:.4f}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1,
                    cv2.LINE_AA)
        y_offset += 22

    return image


def _unit_vec(v, eps=BONE_EPS):
    """Unit vector of a 2D bone vector; zero-length/non-finite → (0, 0).

    Never carries a previous frame's direction: keeps extract() a pure
    per-frame function (deterministic, testable, no temporal leakage).
    """
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps or not np.isfinite(n):
        return np.zeros(2, dtype=np.float64)
    return v / n


def _get_joint(lm, idx, vis_thresh):
    """Return (x, y, vis, z) of pose landmark idx if trusted, else None."""
    try:
        res = lm[idx]
    except IndexError:
        return None
    if res is None:
        return None
    if getattr(res, 'visibility', 1.0) < vis_thresh:
        return None
    return (float(res.x), float(res.y), float(getattr(res, 'visibility', 1.0)),
            float(getattr(res, 'z', 0.0)))


def _retarget_arm(shoulder_canon, upper_raw, fore_raw):
    """FK: elbow = shoulder + unit(upper)*STD_UPPER_ARM; wrist = elbow + unit(fore)*STD_FOREARM."""
    elbow = shoulder_canon + _unit_vec(upper_raw) * STD_UPPER_ARM
    wrist = elbow + _unit_vec(fore_raw) * STD_FOREARM
    return elbow, wrist


def _process_hand(hand_lm):
    """Normalize one hand around its own wrist landmark; missing/degenerate → zeros(42).

    palm_size = max(|wrist0 − mcp9|, |mcp5 − mcp17| × 1.25) in raw image units:
    the max-branch keeps scale stable when the hand foreshortens toward the
    camera (one axis collapses, the other survives). Points are translated to
    the hand detector's own wrist (pts[0]), divided by palm_size and scaled by
    STD_PALM. Purely local — no global canonical wrist is added (the global
    wrist already lives in the pose slots).
    """
    if hand_lm is None:
        return np.zeros(HAND_DIM)
    pts = np.array([(res.x, res.y) for res in hand_lm.landmark], dtype=np.float64)
    if pts.shape != (21, 2):
        return np.zeros(HAND_DIM)
    w0, mcp5, mcp9, mcp17 = pts[0], pts[5], pts[9], pts[17]
    palm_size = max(float(np.linalg.norm(w0 - mcp9)),
                    float(np.linalg.norm(mcp5 - mcp17)) * 1.25)
    if palm_size < MIN_PALM_SIZE or not np.isfinite(palm_size):
        return np.zeros(HAND_DIM)
    normed = (pts - w0) / palm_size * STD_PALM
    return normed.flatten()


def _palm_size(hand_lm):
    """Palm scale in raw image units, or None if the hand is missing/degenerate.

    Shared by _process_hand and _face_offsets so both normalize with the
    same per-hand scale.
    """
    if hand_lm is None:
        return None
    pts = np.array([(res.x, res.y) for res in hand_lm.landmark], dtype=np.float64)
    if pts.shape != (21, 2):
        return None
    palm_size = max(float(np.linalg.norm(pts[0] - pts[9])),
                    float(np.linalg.norm(pts[5] - pts[17])) * 1.25)
    if palm_size < MIN_PALM_SIZE or not np.isfinite(palm_size):
        return None
    return palm_size


def _get_face_center(results, vis_thresh):
    """Face center in normalized (x, y) coords, or None if unavailable.

    Primary: midpoint of the two pose ears (7, 8) — same coordinate space as
    the hands, with visibility gating. Fallback: midpoint of the face-mesh
    outer eye corners (33, 263) when pose ears are missing but the face mesh
    is present.
    """
    if results.pose_landmarks is not None:
        lm = results.pose_landmarks.landmark
        le = _get_joint(lm, L_EAR, vis_thresh)
        re = _get_joint(lm, R_EAR, vis_thresh)
        if le is not None and re is not None:
            return ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    face = getattr(results, 'face_landmarks', None)
    if face is not None:
        try:
            lo, ro = face.landmark[L_EYE_OUTER], face.landmark[R_EYE_OUTER]
            return ((float(lo.x) + float(ro.x)) / 2.0,
                    (float(lo.y) + float(ro.y)) / 2.0)
        except (IndexError, AttributeError):
            return None
    return None


def _get_raised_shoulders(results, vis_thresh):
    """Raised shoulder anchors in normalized (x, y): (ls_raised, rs_raised, mid).

    Keeps each shoulder's x, lifts y to the face-center y from
    _get_face_center (same anchor draw_landmarks_test visualizes).
    Returns (None, None, None) if shoulders or face center are missing.
    """
    ls, rs = _get_shoulders(results)
    if ls is None or rs is None:
        return None, None, None
    fc = _get_face_center(results, vis_thresh)
    if fc is None:
        return None, None, None
    ls_r = (float(ls.x), float(fc[1]))
    rs_r = (float(rs.x), float(fc[1]))
    mid = ((ls_r[0] + rs_r[0]) / 2.0, (ls_r[1] + rs_r[1]) / 2.0)
    return ls_r, rs_r, mid


def _face_offsets(hand_lm, face_center, palm_size):
    """Per-point hand→face-center offsets in canonical palm units.

    offset_i = (pts_i - face_center) / palm_size * STD_PALM, flattened to 42.
    Missing hand, missing face center, or degenerate palm → zeros(42).

    Legacy: kept for data migration / tests; extract() no longer uses it.
    """
    if hand_lm is None or face_center is None or palm_size is None:
        return np.zeros(FACE_REF_DIM)
    pts = np.array([(res.x, res.y) for res in hand_lm.landmark], dtype=np.float64)
    if pts.shape != (21, 2) or not np.all(np.isfinite(pts)):
        return np.zeros(FACE_REF_DIM)
    fc = np.array(face_center, dtype=np.float64)
    return ((pts - fc) / palm_size * STD_PALM).flatten()


def _shoulder_offsets(hand_lm, anchors, palm_size):
    """Per-point hand→anchor offsets for 3 raised-shoulder anchors.

    anchors = (ls_raised, rs_raised, mid); per anchor computes
    (pts - anchor) / palm_size * STD_PALM flattened to 42, concatenated
    in anchor order → 126. Missing hand/anchors/degenerate palm → zeros(126).
    """
    if hand_lm is None or anchors is None or palm_size is None:
        return np.zeros(SH_OFF_REF_DIM)
    if any(a is None for a in anchors):
        return np.zeros(SH_OFF_REF_DIM)
    pts = np.array([(res.x, res.y) for res in hand_lm.landmark], dtype=np.float64)
    if pts.shape != (21, 2) or not np.all(np.isfinite(pts)):
        return np.zeros(SH_OFF_REF_DIM)
    blocks = []
    for a in anchors:
        av = np.array(a, dtype=np.float64)
        blocks.append(((pts - av) / palm_size * STD_PALM).flatten())
    return np.concatenate(blocks)

def _ac_dist(p, q, frame_shape):
    """Euclidean distance in aspect-corrected centered coords.

    Uses aspect_correct() on both endpoints when frame_shape is given;
    falls back to raw normalized hypot when frame_shape is None.
    """
    if frame_shape is None:
        return math.hypot(q[0] - p[0], q[1] - p[1])
    ax, ay = aspect_correct(p[0], p[1], frame_shape)
    bx, by = aspect_correct(q[0], q[1], frame_shape)
    return math.hypot(bx - ax, by - ay)


def _compute_arm_depth_ratio(shoulder, elbow, wrist, frame_shape=None,
                               neutral_len=None, dz=0.0, z_gain=None):
    """Compute foreshortening depth ratio against the arm's neutral length.
    - Forward to camera: 2D distance be shortened -> ratio < 1.0
    - Background to camera: 2D distance be stretched / reaches maximum length -> ratio > 1.0

    Lengths are measured in aspect-corrected normalized units (see
    aspect_correct/aspect-corrected _ac_dist), so the ratio is comparable
    across cameras with different resolutions and aspect ratios.
    neutral_len is the per-session calibrated neutral arm length in the same
    units; falls back to STD_UPPER_ARM + STD_FOREARM for offline data without
    calibration. dz (wrist.z - shoulder.z, MediaPipe relative depth) applies
    a clipped auxiliary correction via z_gain.
    """
    import config as _cfg
    if z_gain is None:
        z_gain = float(getattr(_cfg, 'ARM_DEPTH_Z_GAIN', 2.0))
    upper_raw_len = _ac_dist(shoulder, elbow, frame_shape)
    fore_raw_len = _ac_dist(elbow, wrist, frame_shape)

    actual_total_len = upper_raw_len + fore_raw_len
    standard_total_len = (neutral_len if neutral_len else
                          STD_UPPER_ARM + STD_FOREARM)

    if standard_total_len < 1e-6:
        return 1.0
    ratio = float(actual_total_len/standard_total_len)
    if dz:
        # MediaPipe z grows away from camera: wrist pushed toward camera
        # gives dz < 0, which must shrink the ratio (< 1.0 = toward camera).
        ratio += z_gain * float(dz)
    return float(min(2.0, max(0.3, ratio)))


class ArmDepthCalibrator:
    """Per-session neutral arm-length calibration (one instance per video/session).

    Collects aspect-corrected normalized arm lengths from non-idle frames,
    then reports the median as the neutral length for
    _compute_arm_depth_ratio. Values outside [1.5, 4.0] x shoulder width
    (same units) are rejected as unreliable. Unit-free across cameras.
    """

    def __init__(self, warmup_frames=20):
        self.warmup_frames = warmup_frames
        self._samples = [[], []]

    def update(self, side, arm_len, shoulder_width_ac=None):
        """Record one arm-length sample; returns True once calibrated."""
        if shoulder_width_ac:
            lo, hi = 1.5 * shoulder_width_ac, 4.0 * shoulder_width_ac
            if not (lo <= arm_len <= hi):
                return self.calibrated(side)
        self._samples[side].append(float(arm_len))
        return self.calibrated(side)

    def calibrated(self, side):
        return len(self._samples[side]) >= self.warmup_frames

    def get(self, side, fallback=None):
        if not self._samples[side]:
            return fallback
        return float(np.median(self._samples[side]))


class ArmDepthSmoother:
    """EMA + median-window filter for the 2 arm_depths channels.

    Stateful per-session helper for realtime/collect callers. Training data
    stays raw; apply this only on the live extract() output.
    """

    def __init__(self, alpha=0.4, window=5):
        self.alpha = alpha
        self.window = window
        self._hist = [[], []]
        self._ema = [None, None]

    def update(self, side, value):
        hist = self._hist[side]
        hist.append(float(value))
        if len(hist) > self.window:
            hist.pop(0)
        med = float(np.median(hist))
        if self._ema[side] is None:
            self._ema[side] = med
        else:
            self._ema[side] += self.alpha * (med - self._ema[side])
        return float(self._ema[side])

    def reset(self, side=None):
        sides = [0, 1] if side is None else [side]
        for s in sides:
            self._hist[s] = []
            self._ema[s] = None


def feed_calibrator(results, frame_shape, calibrator, vis_thresh=None):
    """Feed one frame's arm lengths (aspect-corrected units) into calibrator."""
    import config as _cfg
    if vis_thresh is None:
        vis_thresh = getattr(_cfg, 'SHOULDER_VIS_THRESHOLD', 0.5)
    if results.pose_landmarks is None:
        return (False, False)
    lm = results.pose_landmarks.landmark
    ls, rs = _get_shoulders(results)
    sw_ac = _ac_dist((ls.x, ls.y), (rs.x, rs.y), frame_shape) if ls and rs else None
    out = []
    for shoulder_idx, elbow_idx, wrist_idx, side in (
            (L_SHOULDER, L_ELBOW, L_WRIST, 0),
            (R_SHOULDER, R_ELBOW, R_WRIST, 1)):
        sh = _get_joint(lm, shoulder_idx, vis_thresh)
        el = _get_joint(lm, elbow_idx, vis_thresh)
        wr = _get_joint(lm, wrist_idx, vis_thresh)
        if sh is None or el is None or wr is None:
            out.append(calibrator.calibrated(side))
            continue
        arm_ac = (_ac_dist(sh, el, frame_shape)
                  + _ac_dist(el, wr, frame_shape))
        out.append(calibrator.update(side, arm_ac, sw_ac))
    return tuple(out)

def extract(results, frame_shape=None, neutral_lens=None):
    """Canonical-skeleton retargeted keypoints as a 1-D array (356 values).

    Layout: pose 6×(x,y,visibility)=18 [0:18] | lh 21×(x,y)=42 [18:60]
    | rh 42 [60:102] | arm_depths 2 [102:104] | sh_off_lh 126 [104:230]
    | sh_off_rh 126 [230:356].
    Pose in per-side blocks: left [0:9] = shoulder (11),
    elbow (13), wrist (15); right [9:18] = shoulder (12), elbow (14),
    wrist (16); face and torso/lower body dropped, z dropped from pose slots
    (kept only as a relative dz correction inside arm_depths), visibility
    kept raw. arm_depths: foreshortening ratio per arm (actual/neutral
    bone lengths), default 1.0 when side is missing/gated.

    Invariants: shoulders are pinned at (∓0.5, 0); elbows/wrists are rebuilt
    by Forward Kinematics from unit bone vectors with standard lengths
    (1.2 / 1.0), so arm length, shoulder width and camera distance cancel out
    while motion direction is preserved — a stationary arm yields a stable
    non-zero pose. Each hand is normalized by its own palm_size around the
    hand detector's own wrist landmark, independent of the shoulders and of
    the global canonical wrist (which lives only in the pose slots).

    Fallbacks (per-side): shoulders missing or visibility < threshold →
    whole frame zeros (Masking skips). Elbow/wrist missing or untrusted on
    one side → that side's 2 pose joints + its 42 hand slots + its 126
    shoulder-offset slots are zeros, the other side is kept. Missing hand,
    degenerate palm, or missing raised-shoulder anchors → that side's 126
    shoulder-offset slots are zeros, frame otherwise intact.
    """
    import config as _cfg
    vis_thresh = getattr(_cfg, 'SHOULDER_VIS_THRESHOLD', 0.5)
    idle_y_max = getattr(_cfg, 'IDLE_WRIST_Y_MAX', IDLE_WRIST_Y_MAX)
    idle_x_max = getattr(_cfg, 'IDLE_WRIST_X_MAX', IDLE_WRIST_X_MAX)
    if results.pose_landmarks is None:
        return np.zeros(BODY_ONLY_DIM)
    lm = results.pose_landmarks.landmark

    ls = _get_joint(lm, L_SHOULDER, vis_thresh)
    rs = _get_joint(lm, R_SHOULDER, vis_thresh)
    if ls is None or rs is None:
        return np.zeros(BODY_ONLY_DIM)

    pose_out = np.zeros(POSE_DIM)
    lh = np.zeros(HAND_DIM)
    rh = np.zeros(HAND_DIM)
    sh_lh = np.zeros(SH_OFF_REF_DIM)
    sh_rh = np.zeros(SH_OFF_REF_DIM)

    # 2 dimensions for arm depth
    arm_depths = np.ones(2,dtype=np.float64)

    anchors = _get_raised_shoulders(results, vis_thresh)

    sides = [
        (ls, L_ELBOW, L_WRIST, results.left_hand_landmarks, 0, 'lh'),
        (rs, R_ELBOW, R_WRIST, results.right_hand_landmarks, 1, 'rh'),
    ]
    for shoulder, elbow_idx, wrist_idx, hand_lm, side, _name in sides:
        el = _get_joint(lm, elbow_idx, vis_thresh)
        wr = _get_joint(lm, wrist_idx, vis_thresh)
        if el is None or wr is None:
            continue  # side stays zeros
        sx = -STD_HALF_SHOULDER if side == 0 else STD_HALF_SHOULDER
        shoulder_canon = np.array([sx, 0.0])

        # calculate new depth before retarget
        neutral = neutral_lens[side] if neutral_lens else None
        dz = wr[3] - shoulder[3]
        arm_depths[side] = _compute_arm_depth_ratio(
            shoulder, el, wr, frame_shape=frame_shape,
            neutral_len=neutral, dz=dz)

        upper_raw = (el[0] - shoulder[0], el[1] - shoulder[1])
        fore_raw = (wr[0] - el[0], wr[1] - el[1])
        elbow_c, wrist_c = _retarget_arm(shoulder_canon, upper_raw, fore_raw)
        if wrist_c[1] > idle_y_max and abs(wrist_c[0]) < idle_x_max:
            continue  # idle hanging arm → side stays zeros, depth keeps measured value
        base = side * 9
        pose_out[base:base + 3] = [sx, 0.0, shoulder[2]]
        pose_out[base + 3:base + 6] = [elbow_c[0], elbow_c[1], el[2]]
        pose_out[base + 6:base + 9] = [wrist_c[0], wrist_c[1], wr[2]]
        hand = _process_hand(hand_lm)
        sh_off = _shoulder_offsets(hand_lm, anchors, _palm_size(hand_lm))
        if side == 0:
            lh = hand
            sh_lh = sh_off
        else:
            rh = hand
            sh_rh = sh_off
    return np.concatenate([pose_out, lh, rh, arm_depths, sh_lh, sh_rh])



# ── Feature selection ────────────────────────────────────────────────────────
# Layout of the 356-dim vector: pose [0:18] | lh [18:60] | rh [60:102] | arm_depths [102:104]
#   | sh_off_lh [104:230] | sh_off_rh [230:356]
# Pose keeps 6 canonical joints only: L/R shoulder, L/R elbow, L/R wrist.
POSE_DIM = 6 * 3
FACE_DIM = 0  # face landmarks are no longer extracted
HAND_DIM = 21 * 2
ARM_DEPTH_DIM = 2   # left/right arm foreshortening depth ratio
FACE_REF_DIM = 21 * 2  # legacy per-hand hand→face-center offsets (migration/tests)
FACE_OFF_DIM = FACE_REF_DIM * 2  # legacy both-hands face offsets
SH_OFF_REF_DIM = 21 * 2 * 3  # per-hand offsets to 3 raised-shoulder anchors (x, y per point)
SH_OFF_DIM = SH_OFF_REF_DIM * 2  # both hands
FULL_DIM = 1662  # legacy absolute-coordinate format (pre-conversion data)
BODY_ONLY_DIM = POSE_DIM + HAND_DIM * 2 + ARM_DEPTH_DIM + SH_OFF_DIM  # 356


def select_features(vec):
    """Deprecated no-op: extract() already returns the 356-dim feature vector."""
    return vec


def n_features():
    """Feature dim the model should expect given current config."""
    return BODY_ONLY_DIM
