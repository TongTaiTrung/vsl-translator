import { signForLetter, MARK_SIGNS } from "./signMap.js";

// Combining marks after NFD decomposition.
const SHAPE_MARKS = {
  "̂": "mu-circumflex", // U+0302 ^ cho â, ô, ê
  "̆": "mu-breve", // U+0306 ( cho ă
  "̛": "moc", // U+031B ' cho ơ, ư
};
const TONE_MARKS = {
  "́": "sac", // U+0301
  "̀": "huyen", // U+0300
  "̉": "hoi", // U+0309
  "̃": "nga", // U+0303
  "̣": "nang", // U+0323
};

/**
 * text tiếng Việt -> token list.
 * Token: {type:'sign', src, label} | {type:'gap'} | {type:'fallback', label}
 * Chữ có dấu tách thành base + dấu (assets là fingerspelling + dấu rời).
 */
export function transliterate(text) {
  const tokens = [];
  for (const word of String(text || "").toLowerCase().split(/\s+/)) {
    if (!word) continue;
    if (tokens.length) tokens.push({ type: "gap" });
    for (const ch of Array.from(word.normalize("NFD"))) {
      if (/[̀-ͯ]/.test(ch)) {
        const key = SHAPE_MARKS[ch] || TONE_MARKS[ch];
        if (key && MARK_SIGNS[key]) {
          tokens.push({ type: "sign", src: MARK_SIGNS[key], label: key });
        }
        continue;
      }
      const src = signForLetter(ch);
      if (src) {
        tokens.push({ type: "sign", src, label: ch });
      } else {
        tokens.push({ type: "fallback", label: ch });
      }
    }
  }
  return tokens;
}
