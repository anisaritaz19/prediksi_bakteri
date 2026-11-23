"""diagram.py - Diagram Stock-Flow model logistik (digambar dengan matplotlib)."""
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon


def _panah(ax, p1, p2, warna="#1f2937", lw=2.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16, lw=lw,
                                 color=warna, connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))


def _katup(ax, x, y, label, warna="#1f2937", sisi="kiri"):
    """Simbol katup (bow-tie) pada pipa aliran; label di bawah pipa, di samping panah parameter."""
    s = 0.17
    ax.add_patch(Polygon([[x - s, y + s], [x - s, y - s], [x + s, y + s], [x + s, y - s]],
                         closed=True, fc="white", ec=warna, lw=1.8, zorder=3))
    if sisi == "kiri":
        ax.text(x - 0.12, y - 0.4, label, ha="right", va="top", fontsize=10, fontweight="bold", color=warna)
    else:
        ax.text(x + 0.12, y - 0.4, label, ha="left", va="top", fontsize=10, fontweight="bold", color=warna)


def gambar_stock_flow(r=None, K=None, judul="Diagram Stock-Flow Pertumbuhan Populasi Bakteri (Model Logistik)"):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title(judul, fontsize=12.5, fontweight="bold", pad=8)

    # awan sumber & sink
    for (cx, cy) in [(0.8, 3.0), (10.2, 3.0)]:
        ax.add_patch(Circle((cx, cy), 0.33, fc="#e5e7eb", ec="#6b7280", lw=1.5, zorder=2))
        ax.text(cx, cy + 0.5, "awan\n(sumber)" if cx < 5 else "awan\n(sink)", ha="center", va="bottom",
                fontsize=8.5, color="#6b7280")

    # stock
    ax.add_patch(FancyBboxPatch((4.3, 2.35), 2.4, 1.3, boxstyle="round,pad=0.02", fc="#dbeafe",
                                ec="#1d4ed8", lw=2.6, zorder=2))
    ax.text(5.5, 3.0, "STOCK\nP(t)\nPopulasi bakteri", ha="center", va="center", fontsize=11,
            fontweight="bold", color="#1e3a8a", zorder=3)

    # aliran masuk & keluar
    _panah(ax, (1.13, 3.0), (4.3, 3.0), "#15803d", 3.2)
    _panah(ax, (6.7, 3.0), (9.87, 3.0), "#b91c1c", 3.2)
    _katup(ax, 2.7, 3.0, "Inflow: r·P\n(pertumbuhan)", "#15803d", "kiri")
    _katup(ax, 8.3, 3.0, "Outflow: r·P²/K\n(pembatasan kapasitas)", "#b91c1c", "kanan")

    # parameter
    ax.text(2.7, 1.45, "r : laju pertumbuhan", ha="center", fontsize=10, color="#15803d",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f0fdf4", ec="#15803d"))
    ax.text(8.3, 1.45, "K : kapasitas dukung", ha="center", fontsize=10, color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#b91c1c"))
    _panah(ax, (2.7, 1.72), (2.7, 2.75), "#15803d", 1.4, "-|>", 0, "--")
    _panah(ax, (8.3, 1.72), (8.3, 2.75), "#b91c1c", 1.4, "-|>", 0, "--")

    # loop umpan balik (stock -> katup)
    _panah(ax, (4.7, 3.68), (3.0, 3.35), "#15803d", 1.8, "-|>", 0.45, "--")
    _panah(ax, (6.3, 3.68), (8.0, 3.35), "#b91c1c", 1.8, "-|>", -0.45, "--")
    ax.text(3.75, 4.55, "R  (+)\numpan balik\npenguat", ha="center", fontsize=10, fontweight="bold",
            color="#15803d")
    ax.text(7.25, 4.55, "B  (−)\numpan balik\npenyeimbang", ha="center", fontsize=10, fontweight="bold",
            color="#b91c1c")

    pers = r"$\frac{dP}{dt}=\mathrm{Inflow}-\mathrm{Outflow}=rP-\frac{rP^2}{K}=rP\left(1-\frac{P}{K}\right)$"
    ax.text(5.5, 0.55, pers, ha="center", va="center", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f9fafb", ec="#9ca3af"))
    if r is not None and K is not None:
        ax.text(5.5, 0.05, f"Parameter terkalibrasi: r = {r:.5f}   |   K = {K:.4f}", ha="center",
                fontsize=9.5, color="#374151")
    fig.tight_layout()
    return fig
