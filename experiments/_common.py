"""Shared helpers: output paths, plot style, JSON results."""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> results/figures/{name}.png")


def dump(name: str, obj) -> None:
    path = RES / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"  data   -> results/{name}.json")
