"""Analyse benchmark results: average ranks, Friedman test, pairwise Wilcoxon tests, figures.
Usage: python analyze.py --results results --figures figures
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

EXPS = {"A: baseline": "noise0_sub0", "B: +20 noise features": "noise20_sub0", "C: 300-row subsample": "noise0_sub300"}
MODELS = ["LogReg", "RandomForest", "HistGB", "MLP"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()
    Path(a.figures).mkdir(exist_ok=True)

    tables, ranks = {}, {}
    for label, tag in EXPS.items():
        t = pd.read_csv(f"{a.results}/mean_auc_{tag}.csv", index_col=0)[MODELS]
        tables[label] = t
        ranks[label] = t.rank(axis=1, ascending=False).mean()
        print(f"\n=== {label}: mean ROC-AUC per dataset\n{t.round(4)}")
        print(f"\nAverage rank (1 = best):\n{ranks[label].round(2).sort_values()}")
        print(f"Mean AUC over datasets:\n{t.mean().round(4)}")
        stat, pval = friedmanchisquare(*[t[m] for m in MODELS])
        print(f"Friedman: chi2={stat:.3f}, p={pval:.4f}")
        print("Pairwise Wilcoxon signed-rank (two-sided, n=8 datasets, uncorrected):")
        for m1, m2 in combinations(MODELS, 2):
            w = wilcoxon(t[m1], t[m2])
            print(f"  {m1} vs {m2}: p={w.pvalue:.4f}, mean diff={np.mean(t[m1]-t[m2]):+.4f}")
        wins = t.idxmax(axis=1).value_counts()
        print(f"Datasets won:\n{wins}")

    base = tables["A: baseline"]
    print("\n=== Change in mean AUC vs baseline (averaged over datasets)")
    for label in ["B: +20 noise features", "C: 300-row subsample"]:
        d = (tables[label] - base).mean()
        print(f"{label}:\n{d.round(4)}")

    raw = pd.read_csv(f"{a.results}/raw_noise0_sub0.csv")
    print("\nMean fit+search time per outer fold (seconds), baseline:")
    print(raw.groupby("model").seconds.mean().round(1))

    # Figure 1: average ranks
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.25
    for i, (label, r) in enumerate(ranks.items()):
        ax.bar(np.arange(4) + i * w, r[MODELS].values, w, label=label)
    ax.set_xticks(np.arange(4) + w); ax.set_xticklabels(MODELS)
    ax.set_ylabel("Average rank (lower is better)"); ax.legend(fontsize=8)
    ax.set_title("Average rank over 8 datasets")
    fig.tight_layout(); fig.savefig(f"{a.figures}/average_ranks.png", dpi=150); plt.close(fig)

    # Figure 2: AUC change vs baseline
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, label in enumerate(["B: +20 noise features", "C: 300-row subsample"]):
        ax.bar(np.arange(4) + i * 0.35, (tables[label] - base).mean()[MODELS].values, 0.35, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(4) + 0.175); ax.set_xticklabels(MODELS)
    ax.set_ylabel("Change in mean ROC-AUC vs baseline"); ax.legend(fontsize=8)
    ax.set_title("Effect of noise features and small data")
    fig.tight_layout(); fig.savefig(f"{a.figures}/auc_change.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
