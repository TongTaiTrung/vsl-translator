const BASE = "/static/signs";

// Single letters that have an asset file.
const LETTERS = "abcdeghiklmnoopqrstuvxy".split("");

export const LETTER_SIGNS = Object.fromEntries(
  LETTERS.map((ch) => [ch, `${BASE}/${ch}.png`]),
);
LETTER_SIGNS["đ"] = `${BASE}/đ.png`;

// Diacritic marks (dấu phụ) and tones (thanh điệu).
export const MARK_SIGNS = {
  "mu-circumflex": `${BASE}/dau-mu-a-o-e.png`, // ^ cho â, ô, ê
  "mu-breve": `${BASE}/dau-mu-a-bre.png`, // ( cho ă
  "moc": `${BASE}/dau-moc-o-u.png`, // ' cho ơ, ư
  "sac": `${BASE}/thanh-sac.png`,
  "huyen": `${BASE}/thanh-huyen.png`,
  "hoi": `${BASE}/thanh-hoi.png`,
  "nga": `${BASE}/thanh-nga.png`,
  "nang": `${BASE}/thanh-nang.png`,
};

export function signForLetter(ch) {
  return LETTER_SIGNS[ch] || null;
}
