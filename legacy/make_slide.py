"""Generate a slide-ready architecture diagram for PS_Opt_V2.

Output: architecture_slide.png (1920x1080, 16:9, 300 dpi).

Run:
    source venv/bin/activate
    python legacy/make_slide.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent.parent / "architecture_slide.png"

# ---------- palette
BG       = "#ffffff"
INK      = "#1f2937"           # primary stroke / text
SOFT     = "#6b7280"           # secondary text
RULE     = "#d1d5db"           # faint grid
BLUE     = "#2563eb"           # orchestrator / data
GREEN    = "#15803d"           # processing
RED      = "#b91c1c"           # external (Lumerical)
PURPLE   = "#6d28d9"           # storage
AMBER    = "#b45309"           # config
ACCENT   = "#c2410c"           # control-loop highlight

# fills (pale tints of accents)
F_BLUE   = "#eff6ff"
F_GREEN  = "#f0fdf4"
F_RED    = "#fef2f2"
F_PURPLE = "#f5f3ff"
F_AMBER  = "#fffbeb"

mpl.rcParams.update({
    "font.family": "Helvetica Neue",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

W, H = 16, 9
DPI = 200

fig = plt.figure(figsize=(W, H), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.set_facecolor(BG)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])

# ---------- helpers
def line(x0, y0, x1, y1, c=INK, lw=1.2, ls="-", alpha=1.0, z=2):
    ax.add_line(Line2D([x0, x1], [y0, y1], color=c, lw=lw, ls=ls,
                       alpha=alpha, solid_capstyle="round", zorder=z))

def arrow(x0, y0, x1, y1, c=INK, lw=1.3, head=10, ls="-", alpha=1.0):
    sty = f"-|>,head_length={head},head_width={head*0.55}"
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle=sty, color=c, lw=lw,
                                 linestyle=ls, alpha=alpha,
                                 shrinkA=0, shrinkB=0,
                                 mutation_scale=1.0, zorder=4))

def box(cx, cy, w, h, title, sub=None,
        fill=F_BLUE, edge=BLUE, title_c=INK, sub_c=SOFT,
        title_sz=14, sub_sz=10, rounding=0.06):
    x, y = cx - w / 2, cy - h / 2
    rect = mpl.patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        fc=fill, ec=edge, lw=1.6, zorder=3)
    ax.add_patch(rect)
    ty = cy + (0.10 if sub else 0)
    ax.text(cx, ty, title, ha="center", va="center",
            fontsize=title_sz, color=title_c, fontweight="semibold")
    if sub:
        ax.text(cx, cy - 0.16, sub, ha="center", va="center",
                fontsize=sub_sz, color=sub_c)

def cluster(x, y, w, h, label, edge=INK, fill=None):
    rect = mpl.patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.10",
        fc=fill if fill else "none", ec=edge, lw=1.0,
        linestyle=(0, (4, 3)), alpha=0.55, zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.18, y + h - 0.22, label,
            fontsize=10, color=edge, alpha=0.85,
            ha="left", va="top", fontweight="semibold")

# ---------- layout
# Header
ax.text(0.55, 8.45, "PS_Opt_V2", fontsize=28, color=INK,
        fontweight="bold", ha="left", va="center")
ax.text(0.55, 7.95, "Silicon PIN phase-shifter optimization — system architecture",
        fontsize=13, color=SOFT, ha="left", va="center")
line(0.55, 7.65, 15.45, 7.65, c=RULE, lw=1.0)

# ---------- clusters
cluster(0.55, 0.55, 4.30, 6.85, "CONFIG", edge=AMBER, fill=F_AMBER)
cluster(5.10, 0.55, 5.95, 6.85, "PER-SIMULATION PIPELINE  ·  run_simulation.run_row",
        edge=GREEN, fill=F_GREEN)
cluster(11.30, 0.55, 4.15, 6.85, "OPTIMIZATION  ·  STORAGE",
        edge=PURPLE, fill=F_PURPLE)

# ---------- left column — config + orchestrator + sampling
box(2.70, 6.65, 3.80, 0.78, "config.py",
    "bounds · weights · targets · flags",
    fill="#ffffff", edge=AMBER)

box(2.70, 5.40, 3.80, 0.95, "main.py",
    "stage controller  ·  LHS  ›  sims  ›  BO",
    fill="#ffffff", edge=BLUE, title_sz=15)

box(2.70, 3.95, 3.80, 0.88, "LHS.py",
    "Latin Hypercube  ·  smt",
    fill="#ffffff", edge=GREEN)

box(2.70, 2.55, 3.80, 0.88, "params.csv",
    "LHS-generated inputs",
    fill="#ffffff", edge=PURPLE, title_sz=13)

# config → main, main → LHS, LHS → params.csv
arrow(2.70, 6.26, 2.70, 5.91, c=AMBER, ls=(0, (3, 3)))
arrow(2.70, 4.93, 2.70, 4.41, c=BLUE)
arrow(2.70, 3.51, 2.70, 3.01, c=GREEN)

# ---------- center — pipeline (5 stages, stacked)
cx_p = 8.08
pipe_ys = [6.65, 5.55, 4.45, 3.35, 2.25]

stages = [
    ("snap_params_dict",       "discrete grid  ·  h_si  ·  10 nm"),
    ("sim_handler.run_full_simulation",
                               "CHARGE  ›  FDE   via lumapi"),
    ("data_processor",         "process_charge_data  ·  C(V)"),
    ("data_processor",         "process_optical_data  ·  Δn_eff  α  Δφ  Vπ"),
    ("cost.py  ·  FoM",        "valid:  w·(α/T)² + w·(VπL/T)²    fail:  C_BASE+β(π−Δφ)²"),
]
for (title, sub), y in zip(stages, pipe_ys):
    box(cx_p, y, 5.45, 0.84, title, sub,
        fill="#ffffff", edge=GREEN, title_sz=13)

# arrows down the pipeline
for y1, y0 in zip(pipe_ys, pipe_ys[1:]):
    arrow(cx_p, y1 - 0.42, cx_p, y0 + 0.42, c=GREEN)

# external Lumerical to the right of stage 2
box(13.35, 5.55, 3.70, 0.84, "Lumerical",
    "CHARGE + FDE   (lumapi)",
    fill=F_RED, edge=RED, title_sz=13)
arrow(cx_p + 2.72, 5.55, 13.35 - 1.85, 5.55, c=RED, head=9)
arrow(13.35 - 1.85, 5.45, cx_p + 2.72, 5.45, c=RED, head=9)

# params.csv → snap (cross-cluster handoff)
arrow(2.70 + 1.90, 2.55, cx_p - 2.72, 2.55 + 4.10, c=PURPLE,
      ls=(0, (3, 3)))
ax.text((2.70 + 1.90 + cx_p - 2.72) / 2, 4.85,
        "rows", fontsize=9, color=SOFT, ha="center")

# ---------- right column — BO + storage
box(13.35, 6.65, 3.70, 0.95, "BO.py",
    "Gaussian Process + UCB  ·  norm params  ·  −log(cost)  ·  kappa decay",
    fill="#ffffff", edge=ACCENT, title_sz=14)

box(13.35, 4.35, 3.70, 0.84, "result.csv",
    "minimal  ·  BO-facing",
    fill="#ffffff", edge=PURPLE, title_sz=13)

box(13.35, 3.25, 3.70, 0.78, "result_full.csv  ·  raw/  ·  errors.csv",
    "full record  ·  per-sim sweeps  ·  failure log",
    fill="#ffffff", edge=PURPLE, title_sz=11, sub_sz=9)

box(13.35, 1.65, 3.70, 0.80, "results_archive/",
    "warm-start  ·  prepare_initial_data.py",
    fill="#ffffff", edge=SOFT, title_sz=12, sub_sz=9)

# cost → result.csv  (pipeline output)
arrow(cx_p + 2.72, 2.25, 13.35 - 1.85, 4.10, c=PURPLE, head=9)

# result.csv → BO  (training data)
arrow(13.35, 4.77, 13.35, 6.17, c=ACCENT, head=10)
ax.text(13.55, 5.47, "train + register",
        fontsize=10, color=ACCENT, ha="left", va="center")

# BO → snap (suggest next x — control loop)
# route: down-left from BO, across the top of pipeline, down to snap
xa, ya = 13.35 - 1.85, 6.65
xb, yb = cx_p, 7.30           # waypoint above pipeline
xc, yc = cx_p, 7.05            # entry near pipeline top
line(xa, ya, xb, ya, c=ACCENT, lw=1.4)
line(xb, ya, xb, yb, c=ACCENT, lw=1.4)
line(xb, yb, xc, yb, c=ACCENT, lw=1.4)
arrow(xc, yb, xc, 6.65 + 0.42, c=ACCENT, head=10)
ax.text(xb, yb + 0.22, "suggest next x",
        fontsize=10, color=ACCENT, ha="center", va="bottom")

# archive → result.csv (warm-start, dashed)
arrow(13.35, 2.05, 13.35, 3.93, c=SOFT, head=8, ls=(0, (3, 3)))
ax.text(13.55, 2.95, "warm-start",
        fontsize=9, color=SOFT, ha="left", va="center")

# ---------- legend (bottom strip)
ly = 0.20
ax.text(0.55, ly, "Stages:", fontsize=10, color=SOFT,
        ha="left", va="center", fontweight="semibold")

items = [
    ("1  sampling",        GREEN),
    ("2  simulation",      GREEN),
    ("3  processing",      GREEN),
    ("4  cost",            GREEN),
    ("5  BO loop",         ACCENT),
    ("external",           RED),
    ("data store",         PURPLE),
    ("config",             AMBER),
]
x = 1.55
for label_, color in items:
    ax.add_patch(Rectangle((x, ly - 0.10), 0.22, 0.20,
                           fc=color, ec="none", alpha=0.85))
    ax.text(x + 0.32, ly, label_, fontsize=9, color=INK,
            ha="left", va="center")
    x += 1.65

# ---------- save
fig.savefig(OUT, dpi=DPI, facecolor=BG, pad_inches=0)
print(f"Wrote: {OUT}  ({W*DPI}×{H*DPI} px)")
