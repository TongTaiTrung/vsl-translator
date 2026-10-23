"""Video overlay drawing helpers (shared by realtime CLI and frontend stream).

Moved verbatim from scripts/realtime_predict.py so frontend/caller.py can
draw frames without importing the realtime CLI module (which pulls keras).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Max overlay lines for the sentence box at the bottom of the video.
SENTENCE_MAX_LINES = 3

# ── Unicode text rendering (Pillow + TrueType) ──────────────────────────────
# cv2.putText only supports ASCII Hershey fonts, so Vietnamese diacritics
# break. Render text via Pillow instead; OpenCV still draws boxes/video.
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
FONT_SIZE = 22
_FONT_CACHE = {}


def _get_font(size=FONT_SIZE):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_PATH, size)
    return _FONT_CACHE[size]


def put_text_unicode(img, text, org, color=(255, 255, 255), size=FONT_SIZE):
    """Draw UTF-8 text (BGR color) onto a BGR OpenCV frame. Returns img."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    b, g, r = color
    draw.text(org, text, font=_get_font(size), fill=(r, g, b))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def text_width(text, size=FONT_SIZE):
    """Pixel width of text with the overlay font (for greedy wrapping)."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), text, font=_get_font(size))
    return bbox[2] - bbox[0]


def text_height(size=FONT_SIZE):
    """Pixel line height (ascender+descender) of the overlay font."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), "Ag", font=_get_font(size))
    return bbox[3] - bbox[1]


def draw_sentence(img, sentence):
    """Draw the committed sentence in a box at the bottom of the frame.

    Wraps with text_width() to fit the frame width, up to
    SENTENCE_MAX_LINES lines. The newest word is drawn green, the rest white.
    """
    h, w = img.shape[:2]
    margin, pad = 20, 10

    words = list(sentence) if sentence else ["(trống)"]
    # Greedy wrap into lines that fit the frame width.
    lines = [""]
    max_w = w - 2 * (margin + pad)
    for wd in words:
        cand = (lines[-1] + " " + wd).strip()
        tw = text_width(cand)
        if tw <= max_w or not lines[-1]:
            lines[-1] = cand
        elif len(lines) < SENTENCE_MAX_LINES:
            lines.append(wd)
        else:
            break  # box full: older words stay in `sentence`, only display clips
    lines = lines[:SENTENCE_MAX_LINES]

    lh = text_height()
    line_h = lh + 8
    box_h = line_h * len(lines) + 2 * pad
    box_y0 = h - box_h - margin
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, box_y0),
                  (w - margin, h - margin), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    y = box_y0 + pad
    newest = words[-1] if sentence else None
    for line in lines:
        if newest and line.endswith(newest) and line != newest:
            # Split line so the newest word renders green.
            head = line[:len(line) - len(newest)].rstrip()
            put_text_unicode(img, head, (margin + pad, y), (255, 255, 255))
            put_text_unicode(img, newest,
                             (margin + pad + text_width(head + " "), y),
                             (0, 255, 0))
        else:
            color = (0, 255, 0) if line == newest else (255, 255, 255)
            put_text_unicode(img, line, (margin + pad, y), color)
        y += line_h
    return img


def draw_sentence_smart(img, smart_text, pending=False, error=None, y_bottom=None):
    """Draw the LLM-cleaned sentence in a box above the raw sentence box.

    Same greedy-wrap layout as draw_sentence (up to SENTENCE_MAX_LINES lines).
    `y_bottom`: bottom edge of this box; defaults to sitting just above the
    raw sentence box. Shows a status line when pending/error/empty.
    """
    h, w = img.shape[:2]
    margin, pad = 20, 10

    if error:
        lines = [f"(smart lỗi: {error})"]
    elif not smart_text:
        lines = ["(smart: ...)" if pending else "(smart: trống)"]
    else:
        lines = [""]
        max_w = w - 2 * (margin + pad)
        for wd in smart_text.split():
            cand = (lines[-1] + " " + wd).strip()
            tw = text_width(cand)
            if tw <= max_w or not lines[-1]:
                lines[-1] = cand
            elif len(lines) < SENTENCE_MAX_LINES:
                lines.append(wd)
            else:
                break
        lines = lines[:SENTENCE_MAX_LINES]

    title = "Smart:"
    lh = text_height()
    line_h = lh + 8
    box_h = line_h * (len(lines) + 1) + 2 * pad
    if y_bottom is None:
        # Reserve space for the raw box: estimate its height the same way
        # draw_sentence does (assume 1 line) and stack above it.
        raw_box_h = line_h + 2 * pad
        y_bottom = h - raw_box_h - margin - 8
    box_y0 = y_bottom - box_h
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, box_y0),
                  (w - margin, y_bottom), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    y = box_y0 + pad
    put_text_unicode(img, title, (margin + pad, y), (0, 200, 255))
    y += line_h
    for line in lines:
        put_text_unicode(img, line, (margin + pad, y), (255, 255, 255))
        y += line_h
    return img


def draw_confidence_bars(img, actions, probs, pred_idx, bar_x=20, bar_y=80,
                         bar_w=200, bar_h=22, gap=10):
    """Draw a labelled confidence bar for each class."""
    for i, (action, prob) in enumerate(zip(actions, probs)):
        y = bar_y + i * (bar_h + gap)

        # Background track
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h),
                      (50, 50, 50), -1)

        # Filled bar — green for predicted class, blue for others
        fill = int(prob * bar_w)
        color = (0, 220, 0) if (pred_idx >= 0 and i == pred_idx) else (200, 120, 0)
        if fill > 0:
            cv2.rectangle(img, (bar_x, y), (bar_x + fill, y + bar_h), color, -1)

        # Label + percentage
        label = f"{action}: {prob * 100:.1f}%"
        put_text_unicode(img, label, (bar_x + bar_w + 8, y + 2),
                         (255, 255, 255), size=18)
