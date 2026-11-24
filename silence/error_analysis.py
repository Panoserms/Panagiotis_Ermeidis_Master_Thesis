"""Which linguistic features separate the test lines a model gets right from those it gets wrong?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

GROUPS = {"all errors": (0, 1), "missed silences": (1,), "false silences": (0,)}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm correction for testing many features at once."""
    order, adjusted, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def compare(table: pd.DataFrame, features: list[str], y_pred, classes=(0, 1)) -> pd.DataFrame:
    """One row per feature: Welch's t-test (yes/no features: chi-squared), correct against wrong predictions."""
    part = table[table["silence_label"].isin(classes)]
    correct = (part["silence_label"] == pd.Series(y_pred, index=table.index)[part.index])
    rows = []
    for feature in features:
        right, wrong = part.loc[correct, feature], part.loc[~correct, feature]
        if set(table[feature].unique()) <= {0, 1}:
            counts = pd.crosstab(correct, part[feature])
            if counts.shape != (2, 2):
                continue
            p = chi2_contingency(counts)[1]
        else:
            if right.nunique() < 2 and wrong.nunique() < 2:
                continue  # the same value everywhere: nothing to test
            p = ttest_ind(right, wrong, equal_var=False)[1]
        rows.append({"feature": feature, "p": p, "mean_correct": right.mean(), "mean_wrong": wrong.mean(),
                     "in_errors": "higher" if wrong.mean() > right.mean() else "lower"})
    result = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    result["p_holm"] = holm(result["p"].to_numpy())
    return result


def plot(results: dict[str, pd.DataFrame], title: str, top: int = 8):
    """The strongest features of each error group. Dark bars pass the Holm correction."""
    fig, axes = plt.subplots(1, len(results), figsize=(15, 3.9))
    for ax, (name, result) in zip(axes, results.items()):
        rows = result.head(top)[::-1]
        ax.barh(rows["feature"], -np.log10(rows["p"]), color=["#1a1b1c" if p < 0.05 else "#b8bfc6" for p in rows["p_holm"]])
        ax.axvline(-np.log10(0.05), color="#ff5f14", linewidth=1.5)
        ax.set_title(f"{name}: {(result['p'] < 0.05).sum()} of {len(result)} below 0.05, {(result['p_holm'] < 0.05).sum()} after Holm",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("−log10(p); orange line: p = 0.05", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig
