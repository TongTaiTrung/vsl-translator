const $ = (id) => document.getElementById(id);

export function renderSmart(s) {
  const next = s.smart_error
    ? `Không viết được câu. ${s.smart_error}`
    : s.smart
      ? s.smart
      : s.smart_pending
        ? "Đang viết câu..."
        : "Chưa có câu.";
  const el = $("smart");
  if (!el || el.textContent === next) return;
  el.textContent = next;
  el.classList.remove("text-in");
  void el.offsetWidth;
  el.classList.add("text-in");
}

export function renderSmartError(msg) {
  $("smart").textContent = `Không đọc được. ${msg}`;
}
