const $ = (id) => document.getElementById(id);

function renderPredBadges(s) {
  const row = $("pred");
  if (!row) return;
  const words = Array.isArray(s.sentence) ? s.sentence.filter(Boolean) : [];
  const live = s.inferring && s.pred && words[words.length - 1] !== s.pred ? s.pred : "";
  const items = words.map((text) => ({ text, kind: "c" }));
  if (live) items.push({ text: live, kind: "l" });
  if (!items.length) items.push({ text: "chưa có", kind: "e" });
  const key = items.map((it) => `${it.kind}:${it.text}`).join("|");
  if (row.dataset.key === key) return;
  const prev = new Set((row.dataset.key || "").split("|").filter(Boolean));
  row.dataset.key = key;
  row.replaceChildren();
  for (const it of items) {
    const el = badge(it.text, it.kind);
    if (!prev.has(`${it.kind}:${it.text}`)) el.classList.add("badge-in");
    row.appendChild(el);
  }
}

function badge(text, kind) {
  const el = document.createElement("span");
  el.className = kind === "e" ? "badge badge-empty" : kind === "l" ? "badge badge-live" : "badge";
  el.textContent = text;
  return el;
}

function bufferFill(buffer) {
  const match = /^(\d+)\/(\d+)$/.exec(String(buffer || ""));
  if (!match || Number(match[2]) <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((Number(match[1]) / Number(match[2])) * 100)));
}

export function renderStatus(s) {
  $("status").textContent = s.status || "-";
  $("buffer").textContent = s.buffer || "-";
  renderPredBadges(s);
  const pct = Math.max(0, Math.min(100, Math.round((Number(s.conf) || 0) * 100)));
  const ring = $("confRing");
  const confPct = $("confPct");
  const bar = $("bufferBar");
  if (ring) ring.style.setProperty("--p", String(pct));
  if (confPct) confPct.textContent = `${pct}%`;
  if (bar) bar.style.width = `${bufferFill(s.buffer)}%`;
  if (s.groq_model) $("groqModel").textContent = s.groq_model;
  if ("groq_configured" in s) {
    $("groqStatus").textContent = s.groq_configured
      ? "đã cấu hình"
      : "chưa có — smart sẽ báo lỗi";
  }
  $("startBtn").disabled = !!s.inferring;
  $("stopBtn").disabled = !s.inferring;
}
