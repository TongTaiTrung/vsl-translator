import { store } from "../store/store.js";

let current = null;

export function stop() {
  if (current) {
    current.pause();
    current = null;
  }
}

export async function speakText(text) {
  text = (text || "").trim();
  if (!text) return;
  stop();
  const res = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data.error) msg = data.error;
    } catch { /* binary hoặc rỗng */ }
    throw new Error(msg);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  current = new Audio(url);
  current.onended = () => URL.revokeObjectURL(url);
  await current.play();
}

// Shared readers: mọi nút đọc gọi các hàm này, không tự đọc DOM/store.
export const readSentence = () =>
  ((store.state.sentence || []).join(" ")).trim();

export const readSmart = () => (store.state.smart || "").trim();

export const readManualInput = () =>
  (document.getElementById("signInput")?.value || "").trim();

export const speakSentence = () => speakText(readSentence());
export const speakSmart = () => speakText(readSmart());
export const speakManualInput = () => speakText(readManualInput());
