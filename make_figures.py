"""Draws the paper's figures from experiments/. Standard library only, plus Edge or Chrome to turn the SVGs into PNGs.

    python paper/make_figures.py     (run from the repo root; writes paper/figures/)
"""
import collections
import os
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

from make_tables import E4, O1, SINGLES, steps, summary

OUT = Path(__file__).resolve().parent / "figures"
FONT = "'Segoe UI', system-ui, sans-serif"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BROWSERS = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def text(x, y, s, size=13, fill=INK2, anchor="start", weight=400, halo=False):
    back = f' paint-order="stroke" stroke="{SURFACE}" stroke-width="5"' if halo else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{back}>{escape(s)}</text>'


def line(x1, y1, x2, y2, stroke=GRID):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="1"/>'


def dot(x, y, fill):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" stroke="{SURFACE}" stroke-width="2"/>'


def frame(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}" role="img" aria-label="{escape(title)}">'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>{"".join(body)}</svg>')


def legend(x, y, items, swatch):
    out = []
    for label, color in items:
        out.append(swatch(x + 6, y - 4, color))
        out.append(text(x + 18, y, label))
        x += 40 + len(label) * 6.2
    return out


def function_hit_vs_score():
    """One dot per E4B laptop run. Runs with identical values sit side by side."""
    groups = [("Off", E4["off"], BLUE), ("On", E4["on"], AQUA), ("O1: location given", O1, ORANGE),
              ("Other arms (skill, placebo, O2, O3, E8)",
               E4["skill"] + E4["placebo"] + [r for k, v in SINGLES.items() for r in v if k.split()[0] in ("O2", "O3", "E8")], MUTED)]
    w, h, left, right, top, bottom = 760, 470, 66, 28, 112, 58
    xmin, xmax, ymax = 0.45, 0.90, 7
    sx = lambda v: left + (v - xmin) / (xmax - xmin) * (w - left - right)
    sy = lambda v: h - bottom - v / ymax * (h - top - bottom)
    body = [text(24, 34, "O1 reached the right function most often and solved no more", 19, INK, weight=600),
            text(24, 58, "E4B on the laptop: 15 min, 60 calls, no compaction, 20 tasks. One dot per run.", 13)]
    body += legend(24, 86, [(g[0], g[2]) for g in groups], lambda x, y, c: dot(x, y, c))
    for v in range(ymax + 1):
        body.append(line(left, sy(v), w - right, sy(v), AXIS if v == 0 else GRID))
        body.append(text(left - 10, sy(v) + 4, str(v), 12, MUTED, "end"))
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        body.append(text(sx(v), h - bottom + 20, f"{v:.2f}", 12, MUTED, "middle"))
    body.append(text((left + w - right) / 2, h - 14, "Function hit: share of tasks where the gold function was read or edited", 13, INK2, "middle"))
    body.append(f'<text transform="translate(18 {(top + h - bottom) / 2:.1f}) rotate(-90)" font-size="13" fill="{INK2}" text-anchor="middle">Tasks resolved, of 20</text>')
    points = collections.defaultdict(list)
    for name, runs, color in groups:
        for run in runs:
            s = summary(run)
            points[(s["function_hit_rate"], s["n_resolved"])].append((color, name, run))
    dots, xs, ys = [], [], []
    for (fx, score), items in sorted(points.items()):
        for i, (color, name, run) in enumerate(sorted(items, key=lambda it: it[0] == MUTED)):
            cx = sx(fx) + (i - (len(items) - 1) / 2) * 15
            dots.append((color == MUTED, cx, sy(score), color))
            xs += [fx]
            ys += [score]
            if run == O1[0]:
                body.append(text(cx + 12, sy(score) + 4, "O1, first run", 12, INK2, halo=True))
            if run == O1[1]:
                body.append(text(cx + 12, sy(score) + 4, "O1, second run", 12, INK2, halo=True))
    body += [dot(cx, cy, color) for _, cx, cy, color in sorted(dots, key=lambda d: not d[0])]
    n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
    r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    print(f"function hit figure: {n} runs, Pearson r(function hit, resolved) = {r:.2f}")
    return "fig2_function_hit", w, h, body, "O1 reached the right function most often and solved no more"


def tool_results_by_source():
    """Characters of tool results per task, split by the tool that returned them."""
    arms = [("Off", E4["off"]), ("On", E4["on"]), ("On + skill", E4["skill"]), ("Placebo", E4["placebo"]), ("O1", O1)]
    parts = [("File reads", BLUE), ("Graph tools", ORANGE), ("Shell, edits and other", AQUA)]
    w, h, left, right, top = 760, 348, 112, 76, 112
    xmax, row = 60000, 40
    sx = lambda v: left + v / xmax * (w - left - right)
    body = [text(24, 34, "Turning graph tools on doubles the tool output the agent reads", 19, INK, weight=600),
            text(24, 58, "Tool-result characters per task, E4B on the laptop. Half the rise is graph results, half is extra file reads.", 13)]
    body += legend(24, 86, parts, lambda x, y, c: f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" rx="2" fill="{c}"/>')
    base = top + row * len(arms)
    for v in (20000, 40000, 60000):
        body.append(line(sx(v), top - 6, sx(v), base, GRID))
        body.append(text(sx(v), base + 20, f"{v // 1000}K", 12, MUTED, "middle"))
    body.append(text(sx(0), base + 20, "0", 12, MUTED, "middle"))
    for i, (name, runs) in enumerate(arms):
        chars = collections.Counter()
        for run in runs:
            for r in steps(run):
                tool = r["tool_name"]
                chars["File reads" if tool == "read_file" else "Graph tools" if tool in ("search_similar_code", "get_code_neighbors", "get_code_subgraph") else "Shell, edits and other"] += r.get("tool_result_chars") or 0
        n = len(runs) * 20
        y = top + i * row + 8
        body.append(text(24, y + 16, name, 13, INK))
        x = sx(0)
        total = sum(chars.values()) / n
        drawn = [(p, c) for p, c in parts if chars[p] / n >= 1]
        for j, (part, color) in enumerate(drawn):
            value = chars[part] / n
            width = value / xmax * (w - left - right) - (2 if j < len(drawn) - 1 else 0)
            if j == len(drawn) - 1:
                body.append(f'<path d="M{x:.1f} {y} h{width - 4:.1f} a4 4 0 0 1 4 4 v16 a4 4 0 0 1 -4 4 h-{width - 4:.1f} z" fill="{color}"/>')
            else:
                body.append(f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="24" fill="{color}"/>')
            if name in ("Off", "On") and part != "Shell, edits and other":
                body.append(text(x + 8, y + 17, f"{value / 1000:.1f}k", 12, "#ffffff", weight=600))
            x += width + 2
        body.append(text(x + 6, y + 17, f"{total / 1000:.1f}k", 13, INK, weight=600))
        print(f"tool output figure: {name}: total={total:.0f} " + " ".join(f"{p}={chars[p] / n:.0f}" for p, _ in parts))
    body.append(line(sx(0), top - 6, sx(0), base, AXIS))
    return "fig1_tool_output", w, h, body, "Turning graph tools on doubles the tool output the agent reads"


def render(name, w, h, svg_path):
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit("no Edge or Chrome found to render PNGs; the SVGs are written")
    # Edge's crash handler can outlive the screenshot and hold files in the profile, so don't fail on cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        "--force-device-scale-factor=2", f"--window-size={w},{h}", f"--screenshot={(OUT / (name + '.png')).resolve()}",
                        svg_path.resolve().as_uri()], check=True, capture_output=True, timeout=120)


def main():
    OUT.mkdir(exist_ok=True)
    for name, w, h, body, title in (tool_results_by_source(), function_hit_vs_score()):
        path = OUT / f"{name}.svg"
        path.write_text(frame(w, h, body, title), encoding="utf-8")
        render(name, w, h, path)
        print("wrote", path.with_suffix(".png"))


if __name__ == "__main__":
    main()
