/* VSL web client (ES module): MediaPipe loader + capture + API pipeline.
 *
 * Tầng trên của extract.js (thuần toán, port 1:1 vsl2/landmarks.py).
 * Mọi thứ liên quan tới VSL nằm ở đây — file HTML chỉ init và gọi hàm:
 *
 *   import { initMediaPipe, captureSequence, predictSequence, fetchLabels } from './vsl.js';
 *   await initMediaPipe();
 *   const { features, anyHand } = await captureSequence(video, { onProgress });
 *   const result = await predictSequence(features, apiBase);
 *
 * MediaPipe Tasks Vision được import từ CDN ngay trong module này nên HTML
 * không cần biết URL model hay version tasks-vision.
 */

import { PoseLandmarker, HandLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import {
  DIM, SEQUENCE_LEN, createSession, resetSession, extract, mapHands,
} from './extract.js';

const TASKS_WASM =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task';
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let poseLm = null, handLm = null;
const sess = createSession();

export function isReady() {
  return !!(poseLm && handLm);
}

/** Tải PoseLandmarker + HandLandmarker. Gọi 1 lần lúc mở trang. */
export async function initMediaPipe(onProgress) {
  const t0 = performance.now();
  const step = (stage, detail) => onProgress?.({
    stage,
    detail: detail ?? '',
    elapsedMs: Math.round(performance.now() - t0),
  });
  step('wasm-start', TASKS_WASM);
  const vision = await FilesetResolver.forVisionTasks(TASKS_WASM);
  step('wasm-done');
  step('pose-start', POSE_MODEL);
  poseLm = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO', numPoses: 1,
  });
  step('pose-done');
  step('hand-start', HAND_MODEL);
  handLm = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO', numHands: 2,
  });
  step('hand-done');
  step('ready');
  return { poseLm, handLm };
}

/** Extract 1 frame video → vector 356-dim. Cần initMediaPipe() trước. */
export function extractFrame(video, ts = performance.now()) {
  if (!isReady()) throw new Error('MediaPipe chưa init (gọi initMediaPipe trước).');
  if (!video.videoWidth || !video.videoHeight)
    return { vec: new Array(DIM).fill(0), hasHand: false };
  const pose = poseLm.detectForVideo(video, ts).landmarks?.[0] ?? null;
  const handRes = handLm.detectForVideo(video, ts);
  const { lh, rh, hasHand } = mapHands(handRes);
  return {
    vec: extract(pose, lh, rh, [video.videoHeight, video.videoWidth], sess),
    hasHand,
  };
}

/**
 * Chụp đủ SEQUENCE_LEN frames từ video, mỗi frame cách nhau gapMs.
 * Reset calibrator/smoother đầu phiên. onProgress(i, total) để báo UI.
 */
export async function captureSequence(video, { frames = SEQUENCE_LEN, gapMs = 120, onProgress } = {}) {
  if (!isReady()) throw new Error('MediaPipe chưa init (gọi initMediaPipe trước).');
  resetSession(sess);
  const features = [];
  let anyHand = false;
  for (let i = 0; i < frames; i++) {
    onProgress?.(i + 1, frames);
    const { vec, hasHand } = extractFrame(video, performance.now());
    features.push(vec);
    anyHand = anyHand || hasHand;
    await new Promise((r) => setTimeout(r, gapMs));
  }
  return { features, anyHand };
}

/** POST 12 vector đặc trưng lên server. Trả {label, confidence, probs}. */
export async function predictSequence(features, apiBase) {
  const r = await fetch(apiBase.replace(/\/$/, '') + '/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  });
  const j = await r.json();
  if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

/** GET /health → danh sách labels của model đang serve. */
export async function fetchLabels(apiBase) {
  const r = await fetch(apiBase.replace(/\/$/, '') + '/health');
  const j = await r.json();
  return j.labels ?? [];
}

/**
 * Live stream: extract mỗi intervalMs, giữ sliding window SEQUENCE_LEN
 * vector, predict mỗi khi window đầy. Calibrator/smoother chạy liên tục
 * (không reset giữa chừng) như realtime.py phía Python.
 *
 *   const live = startLive(video, { apiBase, onResult, onError });
 *   stopLive(live); // dừng
 *
 * onResult({ label, confidence, probs, anyHand }) mỗi lần predict xong.
 * onFrame(vec, hasHand) mỗi frame (để vẽ overlay nếu muốn).
 * Bỏ qua tick khi predict trước chưa xong (không dồn request).
 */
export function startLive(video, { apiBase, intervalMs = 120, onResult, onError, onFrame } = {}) {
  if (!isReady()) throw new Error('MediaPipe chưa init (gọi initMediaPipe trước).');
  const state = { stopped: false, timer: null, window: [], hands: [], inFlight: false };
  const tick = async () => {
    if (state.stopped) return;
    let vec, hasHand;
    try {
      ({ vec, hasHand } = extractFrame(video));
    } catch (e) {
      onError?.(e);
      return;
    }
    onFrame?.(vec, hasHand);
    state.window.push(vec);
    state.hands.push(hasHand);
    if (state.window.length > SEQUENCE_LEN) {
      state.window.shift();
      state.hands.shift();
    }
    if (state.window.length < SEQUENCE_LEN || state.inFlight) return;
    state.inFlight = true;
    try {
      const j = await predictSequence(state.window, apiBase);
      onResult?.({ ...j, anyHand: state.hands.some(Boolean) });
    } catch (e) {
      onError?.(e);
    } finally {
      state.inFlight = false;
    }
  };
  state.timer = setInterval(tick, intervalMs);
  return state;
}

/** Dừng vòng live do startLive() tạo. */
export function stopLive(state) {
  if (!state) return;
  state.stopped = true;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

export { DIM, SEQUENCE_LEN };
