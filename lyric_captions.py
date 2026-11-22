#!/usr/bin/env python3
"""Burn lyric captions onto a video in the "black box" style.

Style: bold uppercase white text on a solid black rectangle, centred in frame,
hard cuts between lines. All sizes scale with the frame height, so the same
captions file works for 1080p and 4K.

Captions file: one caption per line, "START END TEXT" (seconds), e.g.

    0.0 2.0 Tanah airku
    2.0 4.0 Tidak kulupakan

Blank lines and lines starting with # are ignored.

Usage:
    python3 lyric_captions.py input.mp4 captions.txt output.mp4
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"

# Proportions measured from the reference frame (relative to frame height /
# cap height). Cap height ~2.8% of frame height; box padding is ~1.6x cap
# height left/right and ~0.7x cap height top/bottom.
CAP_HEIGHT_FRAC = 0.028
PAD_X = 1.6
PAD_Y = 0.70
TRACKING = -0.025  # letter spacing, fraction of font size
UPPERCASE = True


def parse_captions(path):
    caps = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                start, end, text = line.split(maxsplit=2)
                caps.append((float(start), float(end), text))
            except ValueError:
                sys.exit(f"{path}:{n}: expected 'START END TEXT', got: {line!r}")
    return caps


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def font_for_cap_height(cap_px):
    # Find the font size whose capital letters are cap_px tall.
    size = int(cap_px * 1.4)
    for _ in range(4):
        font = ImageFont.truetype(FONT, size)
        top, bottom = font.getbbox("H")[1], font.getbbox("H")[3]
        size = max(1, round(size * cap_px / (bottom - top)))
    return ImageFont.truetype(FONT, size)


def render_caption(text, frame_w, frame_h, path):
    if UPPERCASE:
        text = text.upper()
    cap = frame_h * CAP_HEIGHT_FRAC
    font = font_for_cap_height(cap)
    track = TRACKING * font.size

    # Measure text with manual tracking.
    widths = [font.getlength(ch) for ch in text]
    text_w = sum(widths) + track * (len(text) - 1)
    h_top, h_bot = font.getbbox("H")[1], font.getbbox("H")[3]

    pad_x, pad_y = round(cap * PAD_X), round(cap * PAD_Y)
    box_w = round(text_w) + 2 * pad_x
    box_h = round(h_bot - h_top) + 2 * pad_y

    img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0 = (frame_w - box_w) // 2
    y0 = (frame_h - box_h) // 2
    d.rectangle([x0, y0, x0 + box_w - 1, y0 + box_h - 1], fill=(0, 0, 0, 255))

    x = x0 + pad_x
    y = y0 + pad_y - h_top  # put cap top exactly pad_y below box top
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=font, fill=(255, 255, 255, 255))
        x += w + track
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("captions")
    ap.add_argument("output")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="medium")
    args = ap.parse_args()

    caps = parse_captions(args.captions)
    if not caps:
        sys.exit("no captions found")
    w, h = probe(args.input)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", args.input]
        for i, (_, _, text) in enumerate(caps):
            p = os.path.join(tmp, f"cap{i:03d}.png")
            render_caption(text, w, h, p)
            cmd += ["-loop", "1", "-i", p]

        chain, last = [], "0:v"
        for i, (start, end, _) in enumerate(caps):
            out = f"v{i}"
            chain.append(f"[{last}][{i + 1}:v]overlay=0:0:shortest=1:"
                         f"enable='gte(t,{start})*lt(t,{end})'[{out}]")
            last = out
        cmd += ["-filter_complex", ";".join(chain), "-map", f"[{last}]",
                "-map", "0:a?", "-c:a", "copy",
                "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
