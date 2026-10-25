/* VSL on-device feature extraction (ES module, no dependencies).
 *
 * Port 1:1 của vsl2/landmarks.py (hàm extract() + helpers) sang JS để webapp
 * trích xuất landmark ngay trên trình duyệt và POST 12 vector 356-dim lên
 * server. Mọi hằng số và thứ tự concat giữ nguyên bản Python:
 *
 *   pose 6x(x,y,vis)=18 [0:18] | lh 42 [18:60] | rh 42 [60:102]
 *   | arm_depths 2 [102:104] | sh_off_lh 126 [104:230] | sh_off_rh 126 [230:356]
 *
 * Khác biệt đã biết so với Python (ghi chú, không thể khắc phục phía JS):
 *  - PoseLandmarker/HandLandmarker (Tasks Vision) thay cho Holistic: hand
 *    landmarks không có visibility, handedness do HandLandmarker gắn nhãn
 *    độc lập (không có context toàn thân như Holistic).
 *  - Không có face mesh: faceCenter fallback sang mũi (landmark 0) khi tai
 *    thiếu, thay vì midpoint 2 khóe mắt ngoài (33, 263).
 *
 * Cách dùng:
 *   import { createSession, extract, mapHands } from './extract.js';
 *   const sess = createSession();
 *   const { lh, rh } = mapHands(handRes);
 *   const vec = extract(poseLandmarks, lh, rh, [h, w], sess);
 */

export const L_SHOULDER = 11, R_SHOULDER = 12;
export const L_ELBOW = 13, R_ELBOW = 14, L_WRIST = 15, R_WRIST = 16;
export const L_EAR = 7, R_EAR = 8, NOSE = 0;

export const STD_HALF_SHOULDER = 0.5;
export const STD_UPPER_ARM = 1.2, STD_FOREARM = 1.0, STD_PALM = 0.4;
export const BONE_EPS = 1e-6, MIN_PALM_SIZE = 0.012;
export const VIS_THRESH = 0.5;
export const IDLE_Y_MAX = 1.75, IDLE_X_MAX = 0.95; // = IDLE_WRIST_Y/X_MAX
export const Z_GAIN = 2.0; // = ARM_DEPTH_Z_GAIN
export const CAL_WARMUP = 20; // giữ để tương thích; lọc mẫu qua gate vai
export const EMA_ALPHA = 0.4, SMOOTH_WINDOW = 5; // = ARM_DEPTH_EMA_ALPHA/_WINDOW
export const RATIO_MIN = 0.3, RATIO_MAX = 2.0; // = ARM_DEPTH_RATIO_MIN/MAX

export const DIM = 356;
export const SEQUENCE_LEN = 12; // = vsl2/defaults.py SEQUENCE_LEN
const POSE_DIM = 18, HAND_DIM = 42, SH_OFF_REF_DIM = 126;

// ── Helpers (port trực tiếp, cùng tên + cùng ngữ nghĩa) ──

export function getJoint(lm, idx) {
  const p = lm[idx];
  if (!p) return null;
  if ((p.visibility ?? 1.0) < VIS_THRESH) return null;
  return [p.x, p.y, p.visibility ?? 1.0, p.z ?? 0];
}

function unitVec(v) {
  const n = Math.hypot(v[0], v[1]);
  if (n < BONE_EPS || !isFinite(n)) return [0, 0];
  return [v[0] / n, v[1] / n];
}

function handPts(hand) {
  if (!hand || hand.length !== 21) return null;
  return hand.map((p) => [p.x, p.y]);
}

function palmSize(pts) {
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const s = Math.max(d(pts[0], pts[9]), d(pts[5], pts[17]) * 1.25);
  return s < MIN_PALM_SIZE || !isFinite(s) ? null : s;
}

export function processHand(hand) {
  const pts = handPts(hand);
  if (!pts) return new Array(HAND_DIM).fill(0);
  const ps = palmSize(pts);
  if (!ps) return new Array(HAND_DIM).fill(0);
  const out = [];
  for (const p of pts)
    out.push(((p[0] - pts[0][0]) / ps) * STD_PALM, ((p[1] - pts[0][1]) / ps) * STD_PALM);
  return out;
}

export function faceCenter(pose, face) {
  const le = getJoint(pose, L_EAR), re = getJoint(pose, R_EAR);
  if (le && re) return [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2];
  if (face && face[33] && face[263])
    return [(face[33].x + face[263].x) / 2, (face[33].y + face[263].y) / 2];
  const nose = getJoint(pose, NOSE);
  if (nose) return [nose[0], nose[1]];
  return null;
}

export function raisedAnchors(pose, face) {
  const ls = getJoint(pose, L_SHOULDER), rs = getJoint(pose, R_SHOULDER);
  if (!ls || !rs) return null;
  const fc = faceCenter(pose, face);
  if (!fc) return null;
  const a = [ls[0], fc[1]], b = [rs[0], fc[1]];
  return [a, b, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]];
}

export function shoulderOffsets(hand, anchors, ps) {
  if (!hand || !anchors || ps == null) return new Array(SH_OFF_REF_DIM).fill(0);
  const pts = handPts(hand);
  if (!pts || !pts.every((p) => isFinite(p[0]) && isFinite(p[1])))
    return new Array(SH_OFF_REF_DIM).fill(0);
  const out = [];
  for (const a of anchors)
    for (const p of pts)
      out.push(((p[0] - a[0]) / ps) * STD_PALM, ((p[1] - a[1]) / ps) * STD_PALM);
  return out;
}

// _compute_arm_depth_ratio: khoảng cách aspect-corrected + hiệu chỉnh dz.
export function armDepth(sh, el, wr, frameShape, neutral, dz) {
  const ac = (p, q) => {
    const ax = (p[0] - 0.5) * (frameShape[1] / frameShape[0]), ay = p[1] - 0.5;
    const bx = (q[0] - 0.5) * (frameShape[1] / frameShape[0]), by = q[1] - 0.5;
    return Math.hypot(bx - ax, by - ay);
  };
  const actual = ac(sh, el) + ac(el, wr);
  const standard = neutral || STD_UPPER_ARM + STD_FOREARM;
  let ratio = standard < 1e-6 ? 1.0 : actual / standard;
  if (dz) ratio += Z_GAIN * dz;
  return Math.min(RATIO_MAX, Math.max(RATIO_MIN, ratio));
}

// np.median trên mảng đã sort (trung bình 2 giữa khi chẵn).
function medianSorted(v) {
  if (!v.length) return null;
  const n = v.length, mid = n >> 1;
  return n % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

// ── Trạng thái phiên (thay ArmDepthCalibrator + ArmDepthSmoother) ──

export function createSession() {
  return {
    calSamples: [[], []],
    smHist: [[], []],
    smEma: [null, null],
  };
}

export function resetSession(sess) {
  sess.calSamples = [[], []];
  sess.smHist = [[], []];
  sess.smEma = [null, null];
}

function calGet(sess, s) {
  return medianSorted([...sess.calSamples[s]].sort((a, b) => a - b));
}

function calFeed(sess, s, len, shoulderW) {
  // khớp ArmDepthCalibrator.update: loại mẫu ngoài [1.5, 4.0] x vai.
  if (shoulderW) {
    const lo = 1.5 * shoulderW, hi = 4.0 * shoulderW;
    if (!(lo <= len && len <= hi)) return;
  }
  sess.calSamples[s].push(len);
}

function smUpdate(sess, s, v) {
  const h = sess.smHist[s];
  h.push(v);
  if (h.length > SMOOTH_WINDOW) h.shift();
  const med = medianSorted([...h].sort((a, b) => a - b));
  sess.smEma[s] =
    sess.smEma[s] == null ? med : sess.smEma[s] + EMA_ALPHA * (med - sess.smEma[s]);
  return sess.smEma[s];
}

// ── extract(): port của vsl2/landmarks.py extract() ──

export function extract(pose, lh, rh, frameShape, sess) {
  const Z = (n) => new Array(n).fill(0);
  const depths = [1.0, 1.0];
  if (!pose) return Z(DIM);
  const ls = getJoint(pose, L_SHOULDER), rs = getJoint(pose, R_SHOULDER);
  if (!ls || !rs) return Z(DIM);
  const poseOut = Z(POSE_DIM);
  let lhArr = Z(HAND_DIM), rhArr = Z(HAND_DIM);
  let shL = Z(SH_OFF_REF_DIM), shR = Z(SH_OFF_REF_DIM);
  const anchors = raisedAnchors(pose, null);
  const ac = (p, q) => {
    const ax = (p[0] - 0.5) * (frameShape[1] / frameShape[0]), ay = p[1] - 0.5;
    const bx = (q[0] - 0.5) * (frameShape[1] / frameShape[0]), by = q[1] - 0.5;
    return Math.hypot(bx - ax, by - ay);
  };
  const sides = [
    { sh: ls, el: L_ELBOW, wr: L_WRIST, hand: lh, side: 0 },
    { sh: rs, el: R_ELBOW, wr: R_WRIST, hand: rh, side: 1 },
  ];
  for (const { sh, el: ei, wr: wi, hand, side } of sides) {
    const el = getJoint(pose, ei), wr = getJoint(pose, wi);
    if (!el || !wr) continue; // side giữ zeros
    calFeed(sess, side, ac(sh, el) + ac(el, wr), ac(ls, rs));
    const sx = side === 0 ? -STD_HALF_SHOULDER : STD_HALF_SHOULDER;
    depths[side] = armDepth(sh, el, wr, frameShape, calGet(sess, side), wr[3] - sh[3]);
    const up = unitVec([el[0] - sh[0], el[1] - sh[1]]);
    const fo = unitVec([wr[0] - el[0], wr[1] - el[1]]);
    const elC = [sx + up[0] * STD_UPPER_ARM, up[1] * STD_UPPER_ARM];
    const wrC = [elC[0] + fo[0] * STD_FOREARM, elC[1] + fo[1] * STD_FOREARM];
    if (wrC[1] > IDLE_Y_MAX && Math.abs(wrC[0]) < IDLE_X_MAX) continue; // idle arm
    const b = side * 9;
    poseOut[b] = sx; poseOut[b + 1] = 0; poseOut[b + 2] = sh[2];
    poseOut[b + 3] = elC[0]; poseOut[b + 4] = elC[1]; poseOut[b + 5] = el[2];
    poseOut[b + 6] = wrC[0]; poseOut[b + 7] = wrC[1]; poseOut[b + 8] = wr[2];
    const h = processHand(hand);
    const pts = handPts(hand);
    const so = shoulderOffsets(hand, anchors, pts ? palmSize(pts) : null);
    if (side === 0) { lhArr = h; shL = so; } else { rhArr = h; shR = so; }
  }
  depths[0] = smUpdate(sess, 0, depths[0]);
  depths[1] = smUpdate(sess, 1, depths[1]);
  return [...poseOut, ...lhArr, ...rhArr, ...depths, ...shL, ...shR];
}

// ── Gắn nhãn trái/phải cho kết quả HandLandmarker ──

export function mapHands(handRes) {
  const hands = handRes.landmarks ?? [];
  let lh = null, rh = null;
  (handRes.handedness ?? []).forEach((h, i) => {
    const label = h?.[0]?.categoryName ?? h?.[0]?.displayName ?? '';
    if (label === 'Left') lh = hands[i]; else rh = hands[i];
  });
  return { lh, rh, hasHand: !!(lh || rh) };
}
