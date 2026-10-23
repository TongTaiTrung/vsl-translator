import { pollState } from "./api/stateApi.js";
import { startInference, stopInference } from "./api/inferenceApi.js";
import { clearSentence, backspaceWord } from "./api/sentencesApi.js";
import { retrySmart } from "./api/smartApi.js";
import { speakSentence, speakSmart, speakManualInput } from "./api/ttsApi.js";
import { setGroqKey } from "./api/configApi.js";
import { store } from "./store/store.js";
import { renderStatus } from "./ui/statusView.js";
import { renderSentence } from "./ui/sentenceView.js";
import { renderSmart, renderSmartError } from "./ui/smartView.js";
import { initSignInput } from "./sign/signView.js";
import { openCamera, startSending, stopCapture, isLive } from "./ui/cameraCapture.js";
import { showKeyMessage, clearKeyInput, readKeyInput } from "./ui/settingsView.js";

const $ = (id) => document.getElementById(id);

store.subscribe((s) => {
  renderStatus(s);
  renderSentence(s);
  renderSmart(s);
});

initSignInput();

pollState(500, (data) => store.set(data));

$("backspaceBtn").addEventListener("click", async () => {
  try {
    store.set(await backspaceWord());
  } catch (e) { /* giữ state cũ */ }
});
$("clearBtn").addEventListener("click", async () => {
  try {
    store.set(await clearSentence());
  } catch (e) { /* giữ state cũ */ }
});

$("smartRetryBtn").addEventListener("click", async () => {
  const fallback = $("sentence").textContent.split(/\s+/).filter(Boolean);
  const sentence = store.state.sentence.length ? store.state.sentence : fallback;
  try {
    store.set(await retrySmart(sentence));
  } catch (e) {
    renderSmartError(e.message);
  }
});

const HINT_OFF = "Camera đang tắt. Bấm \"Bắt đầu nhận diện\" để mở camera trên thiết bị này.";

function showHint(text) {
  const hint = $("streamHint");
  hint.style.display = "";
  hint.textContent = text;
}

let starting = false;

$("startBtn").addEventListener("click", async () => {
  if (starting) return;
  starting = true;
  const debug = $("debugChk").checked;
  let id = null;
  try {
    id = await openCamera();
    if (!id) return;
    $("streamHint").style.display = "none";
    const data = await startInference(debug);
    if (!isLive(id)) {
      try { await stopInference(); } catch (_) { /* đã dừng */ }
      return;
    }
    store.set(data);
    startSending(
      id,
      (snap) => store.set(snap),
      (msg) => showHint(msg),
    );
  } catch (e) {
    stopCapture();
    try { store.set(await stopInference()); } catch (_) { /* chưa kịp bật */ }
    showHint(e.message || "Không mở được camera");
  } finally {
    starting = false;
  }
});
$("stopBtn").addEventListener("click", async () => {
  stopCapture();
  try {
    store.set(await stopInference());
  } catch (e) { /* giữ state cũ */ }
  showHint(HINT_OFF);
});

$("saveKeyBtn").addEventListener("click", async () => {
  const key = readKeyInput();
  if (!key) { showKeyMessage("Nhập key trước khi lưu."); return; }
  showKeyMessage("Đang lưu...");
  try {
    const data = await setGroqKey(key);
    store.set(data);
    showKeyMessage("Đã lưu key vào backend/.env — có hiệu lực ngay.");
    clearKeyInput();
  } catch (e) {
    showKeyMessage(`(lỗi: ${e.message})`);
  }
});

$("speakSentenceBtn").addEventListener("click", async () => {
  try {
    await speakSentence();
  } catch (e) { /* bỏ qua, giữ state cũ */ }
});
$("speakSmartBtn").addEventListener("click", async () => {
  try {
    await speakSmart();
  } catch (e) {
    renderSmartError(e.message);
  }
});
$("speakInputBtn").addEventListener("click", async () => {
  try {
    await speakManualInput();
  } catch (e) { /* bỏ qua, giữ state cũ */ }
});
