"""SVG toolkit of the three diagram figures (Fig. 1 overview, Fig. 2 training pipeline, Fig. 3 hardware).

Primitives (rect, line, path, text, group), the equation markup (`_{..}` subscript, `^{..}` superscript,
`\\r{..}` upright, `\\b{..}` bold, everything else italic), Tabler icons (icons/, MIT), raster embedding
as base64 PNG, the fish render, and the SVG -> PDF/PNG export (Inkscape if found, else a headless
Chrome/Edge print with the page sized to the figure; under WSL the Windows installations are found).

The diagrams are laid out in millimetres at the printed column width (W) and included unscaled, so a
font size here is the printed size.
"""
from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from _paths import STATIC as IMG_DIR

# ── page (mm). One IEEE column = 245.718 pt (paper_style.COL_PT) ──────────────
W = 245.718 / 72.27 * 25.4          # = \columnwidth, so the SVG is placed unscaled
FS = 9.0 / 72.27 * 25.4             # 9 pt: the paper's label size (paper_style.LABEL_PT)
PT = 25.4 / 72                    # 1 pt in mm
# The diagrams are laid out in millimetres at the final column / text width and included at
# that width, so a font size here IS the printed size. (An earlier version drew them at 7 pt on
# a page 9/7 larger and let LaTeX scale it back, which of course printed 7 pt: a page scaled by
# 1/k shrinks its type by 1/k too. PRINT_SCALE is kept at 1 so the page equals the design.)
PRINT_SCALE = 1.0                 # page / design
FS_MAIN = 9 * PT                  # titles, sub-block headers, signal labels (paper_style.LABEL_PT)
FS_SMALL = 8 * PT                 # everything else: equations, notes, legends, strip labels (TICK_PT, the floor)
SUBSCRIPT = 0.78                  # sub/superscripts relative to their base: 9 pt -> 7 pt, as LaTeX sets them in the plots
ICON_DIR = Path(__file__).resolve().parent / "icons"   # Tabler Icons (MIT), see icons/LICENSE.md

FONT = "'Times New Roman', Times, 'Nimbus Roman', 'Liberation Serif', serif"     # one family, as the plots (paper_style: Times via LaTeX)
INK = "#2B3440"          # text + strokes
INK_SOFT = "#5B6573"     # secondary text
EDGE = 0.25              # box stroke width (mm)
ARROW_W = 0.3

COL_SIGNAL = "#2B3A55"   # theta_m waveform
COL_FILT = "#D98B4A"     # filtered TG parameter
COL_DES = "#8A94A3"      # commanded (desired) parameter
COL_IN, COL_HID, COL_OUT = "#6B8FBF", "#C9CFD8", "#D98B4A"

# ── equation mini-markup → tspans (unit-tested) ───────────────────────────────

def parse_eq(s: str) -> list[tuple[str, str]]:
    """Split markup into (kind, text) runs; kind ∈ {it, up, bold, sub, sup}."""
    out: list[tuple[str, str]] = []
    buf = ""
    i = 0

    def flush():
        nonlocal buf
        if buf:
            out.append(("it", buf))
            buf = ""

    while i < len(s):
        c = s[i]
        if c in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            flush()
            j = s.index("}", i + 2)
            out.append(("sub" if c == "_" else "sup", s[i + 2:j]))
            i = j + 1
        elif c == "\\" and i + 2 < len(s) and s[i + 2] == "{":
            flush()
            kind = {"r": "up", "b": "bold"}[s[i + 1]]
            j = s.index("}", i + 3)
            out.append((kind, s[i + 3:j]))
            i = j + 1
        else:
            buf += c
            i += 1
    flush()
    return out

def eq_tspans(markup: str, fs: float) -> str:
    """Render markup as <tspan>s. Sub/superscripts use dy shifts (portable to Inkscape)."""
    parts = []
    pending_dy = 0.0
    for kind, txt in parse_eq(markup):
        attrs = []
        dy = pending_dy
        pending_dy = 0.0
        if kind == "sub":
            attrs.append(f'font-size="{fs * SUBSCRIPT:.3f}"')
            dy += 0.30 * fs
            pending_dy = -0.30 * fs
        elif kind == "sup":
            attrs.append(f'font-size="{fs * SUBSCRIPT:.3f}"')
            dy += -0.45 * fs
            pending_dy = 0.45 * fs
        elif kind == "up":
            attrs.append('font-style="normal"')
        elif kind == "bold":
            attrs.append('font-style="normal" font-weight="bold"')
        else:
            attrs.append('font-style="italic"')
        if abs(dy) > 1e-9:
            attrs.append(f'dy="{dy:.3f}"')
        parts.append(f"<tspan {' '.join(attrs)}>{escape(txt)}</tspan>")
    return "".join(parts)

# ── raster helpers (unit-tested where pure) ──────────────────────────────────

def white_to_alpha(rgb: np.ndarray, gain: float = 6.0) -> np.ndarray:
    """Alpha channel that fades near-white pixels out (so a render on white can sit on a
    tinted background). Grey/coloured pixels stay opaque."""
    mean = rgb[..., :3].astype(float).mean(-1)
    return np.clip((255.0 - mean) * gain, 0, 255).astype(np.uint8)

def crop_box(shape_hw: tuple[int, int], x0: int, x1: int, aspect: float,
             y_center: float) -> tuple[int, int, int, int]:
    """Crop (left, top, right, bottom) of width x1-x0 and the given w/h aspect, vertically
    centred on y_center and clamped to the image."""
    h_img = shape_hw[0]
    w = x1 - x0
    h = int(round(w / aspect))
    top = int(round(y_center - h / 2))
    top = max(0, min(top, h_img - h))
    return x0, top, x1, top + h

def png_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def load_fish(path: Path, width_px: int = 1000) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).copy()
    a[..., 3] = white_to_alpha(a)
    im = Image.fromarray(a)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS)

def load_panel(path: Path, aspect: float, width_px: int = 1400
               ) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop the overhead still to `aspect` (a band centred on the camera's centre row,
    where the start->target line lies) and downsample. Returns (image, crop box)."""
    im = Image.open(path).convert("RGB")
    box = crop_box((im.height, im.width), 40, im.width - 40, aspect, y_center=im.height // 2)
    im = im.crop(box)
    return im.resize((width_px, round(width_px * im.height / im.width)), Image.LANCZOS), box

# ── SVG primitives ────────────────────────────────────────────────────────────

def f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")

def text(x, y, s, fs=FS_SMALL, anchor="start", weight="normal", style="normal",
         fill=INK, extra=""):
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{f(fs)}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" {extra}>'
            f"{escape(s)}</text>")

def eq(x, y, markup, fs=FS_SMALL, anchor="start", fill=INK):
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{f(fs)}" text-anchor="{anchor}" '
            f'fill="{fill}">{eq_tspans(markup, fs)}</text>')

def rect(x, y, w, h, fill, rx=1.5, stroke=INK, sw=EDGE, extra=""):
    """Rounded rectangle. Pass stroke="none" for a fill-only band."""
    return (f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{f(rx)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" {extra}/>')

def line(x1, y1, x2, y2, stroke=INK, sw=ARROW_W, dash="", marker=True):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if marker else ""
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" '
            f'stroke-width="{f(sw)}" stroke-linecap="butt"{d}{m}/>')

def path(d, stroke=INK, sw=ARROW_W, dash="", marker=False, fill="none"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if marker else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}{m}/>')

def image(x, y, w, h, uri, extra=""):
    return (f'<image x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" '
            f'preserveAspectRatio="none" xlink:href="{uri}" {extra}/>')

def icon_paths(svg_text: str) -> list[str]:
    """Path 'd' strings of a Tabler outline icon, minus its invisible 24x24 bounding box."""
    import re
    ds = re.findall(r'<path[^>]*\bd="([^"]+)"', svg_text)
    return [d for d in ds if d.replace(" ", "") not in ("M00h24v24H0z", "M0,0h24v24H0z")]

def icon(name: str, x: float, y: float, size: float, color: str = INK, sw: float = 1.8) -> str:
    """Place a 24x24 Tabler outline icon with its top-left corner at (x, y), `size` mm wide."""
    ds = icon_paths((ICON_DIR / f"{name}.svg").read_text(encoding="utf-8"))
    s = size / 24.0
    body = "".join(f'<path d="{d}"/>' for d in ds)
    return (f'<g transform="translate({f(x)},{f(y)}) scale({s:.4f})" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'inkscape:label="icon:{name}">{body}</g>')

def group(label: str, *children: str) -> str:
    gid = label.lower().replace(" ", "_").replace("/", "_")
    body = "\n    ".join(c for c in children if c)
    return f'<g id="{gid}" inkscape:label="{label}">\n    {body}\n  </g>'

def fish_top(cx, cy, length, heading_deg=0.0, color=INK):
    """Tiny top-view fish (ellipse body + triangular tail) centred at (cx, cy), `length` mm long,
    heading `heading_deg` clockwise from up (90 = heading right)."""
    b, w = length * 0.62, length * 0.30
    d = (f"M {cx:.3f},{cy - b / 2:.3f} "
         f"C {cx + w / 2:.3f},{cy - b / 2:.3f} {cx + w / 2:.3f},{cy + b / 2:.3f} {cx:.3f},{cy + b / 2:.3f} "
         f"C {cx - w / 2:.3f},{cy + b / 2:.3f} {cx - w / 2:.3f},{cy - b / 2:.3f} {cx:.3f},{cy - b / 2:.3f} Z "
         f"M {cx:.3f},{cy + b / 2 - 0.1:.3f} L {cx - w * 0.75:.3f},{cy + length / 2:.3f} "
         f"L {cx + w * 0.75:.3f},{cy + length / 2:.3f} Z")
    return (f'<g transform="rotate({f(heading_deg)} {f(cx)} {f(cy)})">'
            f'<path d="{d}" fill="{color}" stroke="{color}" stroke-width="0.15" stroke-linejoin="round"/></g>')

def network(x, y, w, h, layers, in_label="s", out_label="a"):  # noqa: E501
    """Small fully-connected MLP glyph with `layers` nodes per column."""
    xs = np.linspace(x, x + w, len(layers))
    spacing = min(h / (max(layers)), 2.0)
    cols = []
    for n, cx in zip(layers, xs):
        ys = y + h / 2 + (np.arange(n) - (n - 1) / 2) * spacing
        cols.append([(cx, cy) for cy in ys])
    out = []
    for a, b in zip(cols[:-1], cols[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                out.append(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" '
                           f'stroke="#9AA5B5" stroke-width="0.08" stroke-opacity="0.7"/>')
    for i, col in enumerate(cols):
        fill = COL_IN if i == 0 else COL_OUT if i == len(cols) - 1 else COL_HID
        for (cx, cy) in col:
            out.append(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="0.55" fill="{fill}" '
                       f'stroke="{INK}" stroke-width="0.12"/>')
    out.append(eq(x - 1.2, y + h / 2 + 0.7, in_label, fs=FS_SMALL, anchor="end"))
    out.append(eq(x + w + 1.2, y + h / 2 + 0.7, out_label, fs=FS_SMALL, anchor="start"))
    return "\n    ".join(out)

# ── TG waveform ──────────────────────────────────────────────────────────────

# ── figure ────────────────────────────────────────────────────────────────────

# ── PDF export ────────────────────────────────────────────────────────────────


# ── export: SVG -> PDF / PNG ──────────────────────────────────────────────────
import subprocess
import tempfile

_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]

_INKSCAPE_CANDIDATES = [
    r"C:\Program Files\Inkscape\bin\inkscape.exe",
    r"C:\Program Files\Inkscape\bin\inkscape.com",
    "/usr/bin/inkscape",
    "/mnt/c/Program Files/Inkscape/bin/inkscape.exe",
]

def _find(names, candidates):
    import shutil
    for n in names:
        if shutil.which(n):
            return shutil.which(n)
    for c in candidates:
        if Path(c).is_file():
            return c
    return None

# When the browser or Inkscape is a Windows binary launched from WSL (/mnt/c/...), it needs
# \\wsl.localhost\... paths for everything it reads or writes.
def _is_windows_exe(exe) -> bool:
    return bool(exe) and str(exe).startswith("/mnt/")

def _bpath(p, exe) -> str:
    p = Path(p).resolve()
    if _is_windows_exe(exe):
        import subprocess as _sp
        return _sp.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)

def _uri(p, exe) -> str:
    return "file:///" + _bpath(p, exe).replace("\\", "/") if _is_windows_exe(exe) else Path(p).resolve().as_uri()

def _wrapper_html(svg: Path, w_mm: float, h_mm: float, px_per_mm: float | None = None, chrome=None) -> str:
    if px_per_mm:
        size = f"width:{w_mm * px_per_mm:.0f}px;height:{h_mm * px_per_mm:.0f}px"
    else:
        size = f"width:{w_mm}mm;height:{h_mm}mm"
    return ("<!doctype html><html><head><meta charset='utf-8'><style>"
            f"@page{{size:{w_mm}mm {h_mm}mm;margin:0}} html,body{{margin:0;padding:0;background:#fff}}"
            f"img{{display:block;{size}}}"
            f"</style></head><body><img src='{_uri(svg, chrome)}'></body></html>")

def export_pdf(svg: Path, pdf: Path, w_mm: float, h_mm: float) -> None:
    ink = _find(["inkscape"], _INKSCAPE_CANDIDATES)
    if ink:
        subprocess.run([ink, _bpath(svg, ink), "--export-type=pdf", f"--export-filename={_bpath(pdf, ink)}"],
                       check=True, timeout=300)
        print(f"Saved {pdf}  (Inkscape)")
        return
    chrome = _find(["chrome", "google-chrome", "chromium", "msedge"], _CHROME_CANDIDATES)
    if not chrome:
        print("WARNING: no Inkscape or Chrome found; PDF not written.")
        return
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "print.html"
        wrapper.write_text(_wrapper_html(svg, w_mm, h_mm, chrome=chrome), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={_bpath(pdf, chrome)}", _uri(wrapper, chrome)],
                       check=True, timeout=120, capture_output=True)
    if not pdf.is_file():
        print("WARNING: Chrome did not produce a PDF.")
        return
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
    size = f", page {float(m[1]) * 25.4 / 72:.1f} x {float(m[2]) * 25.4 / 72:.1f} mm" if m else ""
    print(f"Saved {pdf}  (Chrome print{size})")

def export_png(svg: Path, png: Path, w_mm: float, h_mm: float, px_per_mm: float = 8.0) -> None:
    chrome = _find(["chrome", "google-chrome", "chromium", "msedge"], _CHROME_CANDIDATES)
    if not chrome:
        return
    wpx, hpx = round(w_mm * px_per_mm), round(h_mm * px_per_mm)
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "shot.html"
        wrapper.write_text(_wrapper_html(svg, w_mm, h_mm, px_per_mm, chrome=chrome), encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={wpx},{hpx}", f"--screenshot={_bpath(png, chrome)}",
                        _uri(wrapper, chrome)], check=True, timeout=120, capture_output=True)
    if png.is_file():
        print(f"Saved {png}  ({wpx}x{hpx} px preview)")
