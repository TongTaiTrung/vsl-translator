const $ = (id) => document.getElementById(id);

export function renderStatus(s) {
  $("status").textContent = s.status || "-";
  $("buffer").textContent = s.buffer || "-";
  $("pred").textContent = s.pred
    ? `${s.pred} (${(s.conf * 100).toFixed(1)}%)`
    : "(chưa có)";
  if (s.groq_model) $("groqModel").textContent = s.groq_model;
  if ("groq_configured" in s) {
    $("groqStatus").textContent = s.groq_configured
      ? "đã cấu hình"
      : "chưa có — smart sẽ báo lỗi";
  }
  $("startBtn").disabled = !!s.inferring;
  $("stopBtn").disabled = !s.inferring;
}
