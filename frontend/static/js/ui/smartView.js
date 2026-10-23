const $ = (id) => document.getElementById(id);

export function renderSmart(s) {
  $("smart").textContent = s.smart_error
    ? `(smart lỗi: ${s.smart_error})`
    : s.smart
      ? s.smart
      : s.smart_pending
        ? "(smart: ...)"
        : "(smart: trống)";
}

export function renderSmartError(msg) {
  $("smart").textContent = `(smart: ${msg})`;
}
