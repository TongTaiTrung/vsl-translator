const $ = (id) => document.getElementById(id);

export function showKeyMessage(msg) {
  $("keyMsg").textContent = msg;
}

export function clearKeyInput() {
  $("groqKey").value = "";
}

export function readKeyInput() {
  return $("groqKey").value.trim();
}
