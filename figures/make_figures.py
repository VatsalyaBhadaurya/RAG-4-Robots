"""Generate publication-grade figures for the README from REAL measured data.

No invented numbers: retrieval metrics come from r4r.eval.evaluate_retrieval and
heuristic scores from r4r.scoring.score_scene. Uses the scientific-figure-making
house style (palette, spines off, vector-friendly export).

Run:  python figures/make_figures.py   ->  figures/*.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
OUT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.0


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": style.font_size + 2,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def finalize(fig, basename, dpi=300):
    path = os.path.join(OUT, basename + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- data
def retrieval_data():
    from r4r.eval import evaluate_retrieval
    try:
        import ollama
        ollama.embeddings(model="nomic-embed-text", prompt="ping")
        from r4r.embeddings import OllamaEmbedder
        emb = OllamaEmbedder("nomic-embed-text")
    except Exception:
        from r4r.embeddings import HashEmbedder
        emb = HashEmbedder(512)
    return evaluate_retrieval(emb)


def scoring_data():
    from r4r.scene_graph import infer_spatial_edges
    from r4r.scenes import build_tabletop_scene
    from r4r.scoring import score_scene
    g = build_tabletop_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    return score_scene(g)


# ---------------------------------------------------------------- figures
def fig_architecture():
    """Styled pipeline diagram (design schematic)."""
    apply_publication_style(FigureStyle(font_size=12))
    stages = [
        ("Perception\nRGB-D", PALETTE["neutral"], "black"),
        ("3D Scene\nGraph", PALETTE["blue_secondary"], "white"),
        ("GraphRAG\nRetrieval", PALETTE["blue_main"], "white"),
        ("GNN +\nHeuristics", PALETTE["teal"], "white"),
        ("LLM Planner\nllama3.2", PALETTE["violet"], "white"),
        ("Validator\nsafety gate", PALETTE["red_strong"], "white"),
        ("MuJoCo +\nPiper arm", PALETTE["green_3"], "black"),
        ("World Update\n+ Log", PALETTE["neutral"], "black"),
    ]
    n = len(stages)
    bw, bh, gap = 1.72, 1.5, 0.42
    x0 = 0.25
    total = x0 * 2 + n * bw + (n - 1) * gap
    fig, ax = plt.subplots(figsize=(total, 4.6))
    ax.set_xlim(0, total); ax.set_ylim(0, 4.6); ax.axis("off")

    y = 2.65
    centers = []
    prev_right = None
    for i, (label, fc, tc) in enumerate(stages):
        x = x0 + i * (bw + gap)
        box = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.6, edgecolor="black", facecolor=fc)
        ax.add_patch(box)
        ax.text(x + bw / 2, y + bh / 2, label, ha="center", va="center",
                color=tc, fontsize=11.5, fontweight="bold", linespacing=1.3)
        centers.append(x + bw / 2)
        if prev_right is not None:
            ax.add_patch(FancyArrowPatch((prev_right, y + bh / 2), (x, y + bh / 2),
                         arrowstyle="-|>", mutation_scale=15, lw=2, color="#333"))
        prev_right = x + bw

    # feedback loop UNDER the boxes: last stage -> scene graph
    sg_cx, last_cx = centers[1], centers[-1]
    ax.add_patch(FancyArrowPatch((last_cx, y), (sg_cx, y),
                 connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
                 mutation_scale=15, lw=2.2, color=PALETTE["blue_main"], linestyle=(0, (5, 3))))
    ax.text((sg_cx + last_cx) / 2, 0.52,
            "world-state update   ·   retrieval & training feedback",
            ha="center", va="center", fontsize=12, style="italic", color=PALETTE["blue_main"])

    ax.text(x0, 4.25, "RAG-for-Robotics + GraphNet — embodied pipeline",
            fontsize=16, fontweight="bold")
    finalize(fig, "architecture")


def fig_retrieval(rep):
    apply_publication_style(FigureStyle(font_size=15))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    names = [c.name for c in rep.cases]

    # panel A: precision / recall / F1 per case
    ax = axes[0]
    P = [c.precision for c in rep.cases]
    R = [c.recall for c in rep.cases]
    F = [c.f1 for c in rep.cases]
    x = np.arange(len(names)); w = 0.27
    ax.bar(x - w, P, w, label="Precision", color=PALETTE["red_2"], edgecolor="black", linewidth=1.2)
    ax.bar(x, R, w, label="Recall", color=PALETTE["blue_main"], edgecolor="black", linewidth=1.2)
    ax.bar(x + w, F, w, label="F1", color=PALETTE["green_3"], edgecolor="black", linewidth=1.2)
    ax.axhline(rep.mean_recall, ls="--", lw=2, color=PALETTE["blue_main"], alpha=0.7)
    ax.text(len(names) - 0.5, rep.mean_recall + 0.02, f"mean recall {rep.mean_recall:.2f}",
            ha="right", color=PALETTE["blue_main"], fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_title("GraphRAG retrieval quality (per task)")
    ax.legend(ncol=3, loc="upper left", fontsize=12)

    # panel B: token usage GraphRAG vs full-graph
    ax = axes[1]
    gr = [c.token_estimate for c in rep.cases]
    fullv = [c.full_graph_token_estimate for c in rep.cases]
    ax.bar(x - w / 1.6, gr, w * 1.25, label="GraphRAG (subgraph + knowledge)",
           color=PALETTE["blue_main"], edgecolor="black", linewidth=1.2)
    ax.bar(x + w / 1.6, fullv, w * 1.25, label="Full-graph prompting",
           color=PALETTE["neutral"], edgecolor="black", linewidth=1.2, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Estimated prompt tokens")
    ax.set_title(f"Token efficiency  (mean {rep.mean_token_savings:.1f}× smaller)")
    ax.legend(loc="upper right", fontsize=12)

    fig.suptitle(f"Retrieval benchmark — {rep.embedder} on the tabletop scene",
                 fontsize=17, fontweight="bold", y=1.02)
    finalize(fig, "retrieval_benchmark")


def fig_scoring(report):
    apply_publication_style(FigureStyle(font_size=15))
    objs = report.ranking
    cols = ["Reachability", "Graspability", "Clearance\n(1 - collision risk)", "Support\nstability"]
    M = np.array([[report.for_object(o).reachability,
                   report.for_object(o).graspability,
                   1.0 - report.for_object(o).collision_risk,
                   report.for_object(o).support_stability] for o in objs])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=12)
    ax.set_yticks(range(len(objs))); ax.set_yticklabels(objs, fontsize=12)
    for i in range(len(objs)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else "black", fontsize=12, fontweight="bold")
    # grasp decision in its own column on the right (clear of the labels)
    mx = len(cols) - 0.5 + 0.45
    ax.text(mx, -0.75, "safe to\ngrasp?", ha="center", va="center", fontsize=11,
            fontweight="bold", clip_on=False)
    for i, o in enumerate(objs):
        ok = report.for_object(o).graspable
        ax.text(mx, i, "YES" if ok else "no", ha="center", va="center",
                color=PALETTE["green_3"] if ok else PALETTE["red_strong"],
                fontsize=13, fontweight="bold", clip_on=False)
    ax.set_xlim(-0.5, len(cols) - 0.5 + 0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cb.set_label("desirability (higher is better)", fontsize=12)
    ax.set_title("Heuristic object scores", pad=12)
    finalize(fig, "scoring_heatmap")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_architecture()
    fig_retrieval(retrieval_data())
    fig_scoring(scoring_data())


if __name__ == "__main__":
    main()
