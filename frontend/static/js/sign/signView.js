import { transliterate } from "./transliterate.js";
import { startMic, stopMic, pollStt } from "../api/sttApi.js";
import { store } from "../store/store.js";

const $ = (id) => document.getElementById(id);

export function getManualText() {
  return ($("signInput")?.value || "").trim();
}

export function renderSignFromText(text) {
  const grid = $("signGrid");
  const caption = $("signCaption");
  if (!grid) return;
  if (caption) caption.textContent = text ? `“${text}”` : "(nhập text rồi bấm Chuyển đổi)";
  grid.innerHTML = "";
  for (const tok of transliterate(text)) {
    if (tok.type === "gap") {
      const gap = document.createElement("div");
      gap.className = "sign-word-gap";
      grid.appendChild(gap);
      continue;
    }
    if (tok.type === "sign") {
      const fig = document.createElement("figure");
      fig.className = "sign-cell";
      const img = document.createElement("img");
      img.src = tok.src;
      img.alt = tok.label;
      img.loading = "lazy";
      const cap = document.createElement("figcaption");
      cap.textContent = tok.label;
      fig.appendChild(img);
      fig.appendChild(cap);
      grid.appendChild(fig);
    } else {
      const fb = document.createElement("div");
      fb.className = "sign-fallback";
      fb.textContent = tok.label;
      fb.title = "chưa có icon cho ký tự này";
      grid.appendChild(fb);
    }
  }
}

export function initSignInput() {
  $("signConvertBtn")?.addEventListener("click", () => {
    renderSignFromText(getManualText());
  });
  $("signInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") renderSignFromText(getManualText());
  });
  $("signFillBtn")?.addEventListener("click", () => {
    const s = (store.state.smart || "").trim();
    if ($("signInput") && s) {
      $("signInput").value = s;
      renderSignFromText(s);
    }
  });
  initMic();
}

let micTimer = null;

function setMicStatus(msg) {
  const el = $("micStatus");
  if (el) el.textContent = msg;
}

function appendHeard(texts) {
  const input = $("signInput");
  if (!input) return;
  for (const t of texts) {
    const cur = input.value.trim();
    input.value = cur ? `${cur} ${t}` : t;
  }
}

function setMicOn(on) {
  const btn = $("micBtn");
  if (!btn) return;
  btn.textContent = on ? "Dừng" : "Mic";
  btn.classList.toggle("is-on", on);
}

export function initMic() {
  $("micBtn")?.addEventListener("click", async () => {
    if (micTimer) {
      clearInterval(micTimer);
      micTimer = null;
      setMicOn(false);
      setMicStatus("");
      try { await stopMic(); } catch { /* bỏ qua */ }
      return;
    }
    setMicStatus("Đang mở mic...");
    try {
      await startMic();
    } catch (e) {
      setMicStatus(`(lỗi: ${e.message})`);
      return;
    }
    setMicOn(true);
    setMicStatus("Đang nghe...");
    micTimer = setInterval(async () => {
      try {
        const data = await pollStt();
        if (data.error) setMicStatus(`(lỗi: ${data.error})`);
        if (data.texts?.length) appendHeard(data.texts);
      } catch (e) {
        setMicStatus(`(lỗi: ${e.message})`);
      }
    }, 1000);
  });
}
