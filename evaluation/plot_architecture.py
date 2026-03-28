"""
Generate a clean, arxiv-style architecture diagram.
Monochrome palette with minimal colour, clear arrows, no overlap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14)
ax.set_ylim(-0.3, 6.8)
ax.axis("off")

# ─── Palette: light fills, black text, thin borders ───
LIGHT = "#f5f5f5"
MID = "#e0e0e0"
ACCENT = "#d0d0d0"
BORDER = "#333333"
TXT = "#1a1a1a"
SUBTXT = "#555555"
ARROW_C = "#444444"


def box(x, y, w, h, label, sub=None, fill=LIGHT, dash=False):
    style = "round,pad=0.1"
    b = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       facecolor=fill, edgecolor=BORDER,
                       linewidth=1.0, linestyle="--" if dash else "-")
    ax.add_patch(b)
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=TXT, family="serif")
        ax.text(x + w / 2, y + h / 2 - 0.15, sub, ha="center", va="center",
                fontsize=7, color=SUBTXT, family="serif", style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=TXT, family="serif")


def arrow(x1, y1, x2, y2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.0,
                                linestyle="--" if dashed else "-"))


def label(x, y, text, fontsize=7.5, color=SUBTXT):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, family="serif")


# ─── Layout constants ───
h = 0.75
bw = 1.55   # box width
sm = 0.22   # spacing

# ═══════════════════════════════════════════
# Row A (y=3.8): Main pipeline
# ═══════════════════════════════════════════
ya = 3.8

x1 = 0.3
box(x1, ya, bw, h, "ECG Input", "12×5000 @ 500 Hz")
arrow(x1 + bw, ya + h/2, x1 + bw + sm, ya + h/2)

x2 = x1 + bw + sm
box(x2, ya, bw, h, "Preprocessing", "Bandpass + Z-norm")
arrow(x2 + bw, ya + h/2, x2 + bw + sm, ya + h/2)

x3 = x2 + bw + sm
cnn_w = 2.4
box(x3, ya, cnn_w, h, "CNN Encoder", "4× ResConvBlock + SE", fill=MID)
arrow(x3 + cnn_w, ya + h/2, x3 + cnn_w + sm, ya + h/2)

x4 = x3 + cnn_w + sm
box(x4, ya, bw, h, "BiLSTM", "2 layers, d=128", fill=MID)
arrow(x4 + bw, ya + h/2, x4 + bw + sm, ya + h/2)

x5 = x4 + bw + sm
box(x5, ya, bw, h, "Classifier", "GAP → Linear(5)")
arrow(x5 + bw, ya + h/2, x5 + bw + sm, ya + h/2)

x6 = x5 + bw + sm
box(x6, ya, bw, h, "Predictions", "P(NORM), ..., P(HYP)")

# ═══════════════════════════════════════════
# Row B (y=2.1): CNN encoder detail
# ═══════════════════════════════════════════
yb = 1.8
dh = 0.65

# ResConvBlock detail - centred, wider layout
parts = ["Conv1d", "BN", "ReLU", "Conv1d", "BN", "SE", "ReLU"]
pw = 0.7
pgap = 0.18
total_w = len(parts) * pw + (len(parts) - 1) * pgap
detail_left = 7.0 - total_w / 2  # centre in figure

# Dashed box
detail_box = FancyBboxPatch((detail_left - 0.2, yb - 0.2), total_w + 0.4, dh + 0.55,
                             boxstyle="round,pad=0.1", facecolor="none",
                             edgecolor="#999999", linewidth=0.8, linestyle="--")
ax.add_patch(detail_box)

for i, p in enumerate(parts):
    px = detail_left + i * (pw + pgap)
    box(px, yb, pw, dh, p, fill=LIGHT if p != "SE" else ACCENT)
    if i < len(parts) - 1:
        arrow(px + pw, yb + dh/2, px + pw + pgap, yb + dh/2)

# Skip connection arc (below the boxes)
arc_start_x = detail_left + 0.35
arc_end_x = detail_left + len(parts) * (pw + pgap) - pgap - 0.35
arc_y = yb - 0.05
ax.annotate("", xy=(arc_end_x, arc_y),
            xytext=(arc_start_x, arc_y),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8,
                            connectionstyle="arc3,rad=-0.45", linestyle="--"))
label((arc_start_x + arc_end_x) / 2, yb - 0.52, "skip connection", fontsize=6.5, color="#888888")

# Label above detail box
label(7.0, yb + dh + 0.22, "ResConvBlock detail (×4)", fontsize=7)

# Dotted lines from CNN encoder down to detail
arrow(x3 + 0.4, ya, detail_left + 0.3, yb + dh + 0.38, dashed=True)
arrow(x3 + cnn_w - 0.4, ya, arc_end_x - 0.3, yb + dh + 0.38, dashed=True)

# ═══════════════════════════════════════════
# Row C (y=5.5): Interpretability branch
# ═══════════════════════════════════════════
yc = 5.3

# Grad-CAM
gc_x = x3 + 0.3
box(gc_x, yc, bw, h, "Grad-CAM", "Class activation map")
arrow(x3 + cnn_w / 2, ya + h, x3 + cnn_w / 2, yc, dashed=False)
label(x3 + cnn_w / 2 - 0.35, (ya + h + yc) / 2, "gradients", fontsize=6.5)

# LLM
llm_x = gc_x + bw + 0.8
box(llm_x, yc, bw, h, "LLM Module", "Prompt → GPT-5.4")
arrow(gc_x + bw, yc + h/2, llm_x, yc + h/2)

# Predictions → LLM
arrow(x6 + bw/2, ya + h, x6 + bw/2, yc + h/2)
arrow(x6 + bw/2, yc + h/2, llm_x, yc + h/2)
label(x6 + bw/2 + 0.35, (ya + h + yc + h/2) / 2, "probabilities", fontsize=6.5)

# Clinical report
rep_x = llm_x + bw + sm
box(rep_x, yc, bw, h, "Clinical Report", "Natural language")
arrow(llm_x + bw, yc + h/2, rep_x, yc + h/2)

# ═══════════════════════════════════════════
# Row D (y=0.6): Augmentation (training only)
# ═══════════════════════════════════════════
yd = 0.7
box(x1, yd, 2.0, 0.55, "Data Augmentation", "Noise · Scale · Wander · Shift", fill=LIGHT, dash=True)
label(x1 + 1.0, yd + 0.55 + 0.15, "(training only)", fontsize=6.5)
arrow(x1 + 2.0, yd + 0.55/2, x2 + bw/2, ya, dashed=True)

plt.tight_layout()
fig.savefig("report/figures/architecture.pdf", bbox_inches="tight", dpi=150)
print("Saved report/figures/architecture.pdf")
plt.close(fig)
