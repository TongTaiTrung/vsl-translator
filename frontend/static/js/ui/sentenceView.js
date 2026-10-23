const $ = (id) => document.getElementById(id);

export function renderSentence(s) {
  $("sentence").textContent = (s.sentence || []).join(" ");
}
