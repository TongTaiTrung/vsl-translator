import { postFrame } from "../api/framesApi.js";

const MAX_W = 1280;
const INTERVAL_MS = 100;

const $ = (id) => document.getElementById(id);

let session = 0;
let tracks = [];
let frameUrl = null;
const canvas = document.createElement("canvas");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clearDebugFrame() {
  const stage = $("stage");
  const img = $("debugFrame");
  if (stage) stage.classList.remove("debug");
  if (img) img.removeAttribute("src");
  if (frameUrl) URL.revokeObjectURL(frameUrl);
  frameUrl = null;
}

function showDebugFrame(b64) {
  const stage = $("stage");
  const img = $("debugFrame");
  if (!stage || !img || !b64) return;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
  img.src = url;
  stage.classList.add("debug");
  if (frameUrl) URL.revokeObjectURL(frameUrl);
  frameUrl = url;
}

function stopTracks() {
  for (const track of tracks) track.stop();
  tracks = [];
  clearDebugFrame();
  const video = $("camera");
  if (!video) return;
  video.srcObject = null;
  video.classList.remove("live");
}

export function stopCapture() {
  session += 1;
  stopTracks();
}

export function isLive(id) {
  return id !== 0 && id === session;
}

function grab(video) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return Promise.resolve(null);
  const scale = Math.min(1, MAX_W / vw);
  const w = Math.max(1, Math.round(vw * scale));
  const h = Math.max(1, Math.round(vh * scale));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, w, h);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.7);
  });
}

async function loop(id, onState, onError) {
  const video = $("camera");
  while (id === session) {
    const t0 = performance.now();
    let blob = null;
    try {
      blob = await grab(video);
    } catch (e) {
      if (id === session) onError(e.message || "Không chụp được khung hình");
    }
    if (id !== session) return;
    if (blob) {
      try {
        const data = await postFrame(blob);
        if (id === session) {
          if (data.frame) showDebugFrame(data.frame);
          delete data.frame;
          onState(data);
        }
      } catch (e) {
        if (id === session && e.status !== 429 && e.status !== 409) {
          onError(e.message || "Không gửi được khung hình");
        }
      }
    }
    const wait = INTERVAL_MS - (performance.now() - t0);
    if (wait > 0) await sleep(wait);
  }
}

// getUserMedia phải được gọi đồng bộ trong sự kiện bấm — iOS bỏ qua
// lời gọi camera nếu có await mạng đứng trước.
export function openCamera() {
  if (!window.isSecureContext) {
    return Promise.reject(new Error(
      "Điện thoại chỉ mở được camera khi vào bằng HTTPS."));
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return Promise.reject(new Error("Trình duyệt không hỗ trợ camera."));
  }
  const id = session + 1;
  session = id;
  stopTracks();
  const video = $("camera");
  const pending = navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  return pending.then(async (stream) => {
    if (id !== session) {
      stream.getTracks().forEach((t) => t.stop());
      return null;
    }
    tracks = stream.getTracks();
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    if (id !== session) {
      stopTracks();
      return null;
    }
    video.classList.add("live");
    return id;
  }).catch((e) => {
    if (id === session) stopTracks();
    if (e && e.name === "NotAllowedError") {
      throw new Error("Trình duyệt chặn camera. Cho phép quyền rồi bấm lại.");
    }
    throw new Error(e.message || "Không mở được camera");
  });
}

export function startSending(id, onState, onError) {
  if (!isLive(id)) return;
  loop(id, onState, onError);
}
