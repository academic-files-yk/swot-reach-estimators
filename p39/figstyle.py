"""Figure style for journal print (single column 88.9 mm, double column 181 mm).

style()               matplotlib settings for 7-8 pt print size and thin lines
panel(ax, "a", text)  left-aligned "(a) text" panel label
check(fig, name)      reports overlapping text and text outside the canvas
save(fig, name, dir)  checks, then writes vector PDF + 600 dpi PNG + 600 dpi TIFF (LZW)
"""
import os
import matplotlib

MM = 1 / 25.4
COL1 = 88.9 * MM
COL2 = 181.0 * MM
INK, INK2, GRID = "#1a1a1a", "#555555", "#e3e3e3"
# colour-blind-safe categorical palette
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a7974", "#6b4fbb", "#8c564b"]


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8, color=INK, pad=4)


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()
             and id(t) not in hidden and t.get_window_extent(r).width > 0]
    boxes = [(t, t.get_window_extent(r).padded(-0.5)) for t in texts]
    issues = []
    for t, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            issues.append(f"outside canvas: '{t.get_text()[:30]}'")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                issues.append(f"overlap: '{boxes[i][0].get_text()[:25]}' <-> '{boxes[j][0].get_text()[:25]}'")
    print(f"{name}: {len(texts)} text items, {'no issues' if not issues else str(len(issues)) + ' issues'}")
    for s in issues:
        print("   ", s)
    return issues


def save(fig, name, folder):
    import matplotlib.pyplot as plt
    from PIL import Image
    check(fig, name)
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"{name}.pdf"))
    png = os.path.join(folder, f"{name}.png")
    fig.savefig(png, dpi=600)
    Image.open(png).convert("RGB").save(os.path.join(folder, f"{name}.tif"), dpi=(600, 600),
                                        compression="tiff_lzw")
    plt.close(fig)
