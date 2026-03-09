import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIMS_ORDER = [2, 3, 5, 10]
FGROUP_ORDER = ["f1-f5", "f6-f9", "f10-f14", "f15-f19", "f20-f24"]

# Tail / cap definitions (set these to match your paper)
TAIL_T = 10.0          # tail event: relERT > TAIL_T
CAP_T  = 1000.0        # cap event: relERT >= CAP_T  (PAR10-style often uses big numbers)

META_COLS = ["Problem", "Dim", "Instance", "Repetition"]  # columns in pred_df that are not algos

def parse_f_id(problem: str) -> int:
    m = re.match(r"f(\d+)$", str(problem).strip())
    if not m:
        raise ValueError(f"Bad Problem format: {problem!r}, expected like 'f1'")
    return int(m.group(1))

def fgroup_from_fid(fid: int) -> str:
    if 1 <= fid <= 5:  return "f1-f5"
    if 6 <= fid <= 9:  return "f6-f9"
    if 10 <= fid <= 14:return "f10-f14"
    if 15 <= fid <= 19:return "f15-f19"
    if 20 <= fid <= 24:return "f20-f24"
    raise ValueError(fid)

def get_algo_cols(pred_df: pd.DataFrame, meta_cols=META_COLS):
    meta = set(meta_cols)
    algo_cols = [c for c in pred_df.columns if c not in meta]
    if not algo_cols:
        raise ValueError("No algorithm columns found. Check META_COLS.")
    return algo_cols

def prepare_df(pred_df: pd.DataFrame, as_scores, sbs_scores, meta_cols=META_COLS) -> pd.DataFrame:
    pred_df = pred_df.copy()
    n = len(pred_df)
    as_scores = np.asarray(as_scores, dtype=float)
    sbs_scores = np.asarray(sbs_scores, dtype=float)
    if len(as_scores) != n or len(sbs_scores) != n:
        raise ValueError(f"Length mismatch: pred_df={n}, as={len(as_scores)}, sbs={len(sbs_scores)}")

    # attach realised relERTs
    pred_df["as_relert"] = as_scores
    pred_df["sbs_relert"] = sbs_scores

    # fgroup
    pred_df["fid"] = pred_df["Problem"].map(parse_f_id)
    pred_df["fgroup"] = pred_df["fid"].map(fgroup_from_fid)

    # chosen algorithm from composite prediction scores (lower is better)
    algo_cols = get_algo_cols(pred_df, meta_cols=meta_cols)
    score_mat = pred_df[algo_cols].to_numpy(dtype=float)
    idx = np.argmin(score_mat, axis=1)
    pred_df["chosen_algo"] = [algo_cols[i] for i in idx]

    # confidence proxy: margin between best and second best composite score
    part = np.partition(score_mat, kth=1, axis=1)
    pred_df["pred_best"] = part[:, 0]
    pred_df["pred_second"] = part[:, 1]
    pred_df["pred_margin"] = pred_df["pred_second"] - pred_df["pred_best"]

    # event flags
    pred_df["as_tail"] = pred_df["as_relert"] > TAIL_T
    pred_df["sbs_tail"] = pred_df["sbs_relert"] > TAIL_T
    pred_df["as_cap"]  = pred_df["as_relert"] >= CAP_T
    pred_df["sbs_cap"] = pred_df["sbs_relert"] >= CAP_T

    return pred_df

def cell_summary(df: pd.DataFrame, q=0.9) -> pd.DataFrame:
    def agg(g):
        asv = g["as_relert"].to_numpy()
        sbv = g["sbs_relert"].to_numpy()
        out = {
            "n": len(g),
            "as_mean": float(asv.mean()),
            "sbs_mean": float(sbv.mean()),
            "as_q": float(np.quantile(asv, q)),
            "sbs_q": float(np.quantile(sbv, q)),
            "as_tail_rate": float((asv > TAIL_T).mean()),
            "sbs_tail_rate": float((sbv > TAIL_T).mean()),
            "as_cap_rate": float((asv >= CAP_T).mean()),
            "sbs_cap_rate": float((sbv >= CAP_T).mean()),
        }
        out["d_mean"] = out["as_mean"] - out["sbs_mean"]
        out["d_q"] = out["as_q"] - out["sbs_q"]
        out["d_tail_rate"] = out["as_tail_rate"] - out["sbs_tail_rate"]
        out["d_cap_rate"] = out["as_cap_rate"] - out["sbs_cap_rate"]
        return pd.Series(out)

    summ = df.groupby(["fgroup", "Dim"], as_index=False).apply(agg).reset_index(drop=True)
    summ["fgroup"] = pd.Categorical(summ["fgroup"], categories=FGROUP_ORDER, ordered=True)
    summ["Dim"] = pd.Categorical(summ["Dim"], categories=BASE_DIMS_ORDER, ordered=True)
    summ = summ.sort_values(["fgroup", "Dim"]).reset_index(drop=True)
    return summ

def heatmap(summ: pd.DataFrame, value_col: str, title: str, fmt=".2f"):
    pivot = summ.pivot(index="fgroup", columns="Dim", values=value_col)\
                .reindex(index=FGROUP_ORDER, columns=BASE_DIMS_ORDER)
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)

    ax.set_yticks(range(len(FGROUP_ORDER)))
    ax.set_yticklabels(FGROUP_ORDER)
    ax.set_xticks(range(len(BASE_DIMS_ORDER)))
    ax.set_xticklabels([str(d) for d in BASE_DIMS_ORDER])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

def tail_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fg, d), g in df.groupby(["fgroup", "Dim"]):
        as_tail = g["as_tail"].to_numpy()
        sbs_tail = g["sbs_tail"].to_numpy()
        as_cap = g["as_cap"].to_numpy()
        sbs_cap = g["sbs_cap"].to_numpy()

        def ctab(a, b):
            return {
                "a1_b1": int(np.sum(a & b)),
                "a1_b0": int(np.sum(a & (~b))),
                "a0_b1": int(np.sum((~a) & b)),
                "a0_b0": int(np.sum((~a) & (~b))),
            }

        tail_ct = ctab(as_tail, sbs_tail)
        cap_ct = ctab(as_cap, sbs_cap)

        rows.append({
            "fgroup": fg, "Dim": d, "n": len(g),
            **{f"tail_{k}": v for k, v in tail_ct.items()},
            **{f"cap_{k}": v for k, v in cap_ct.items()},
            "tail_rate_as": float(as_tail.mean()),
            "tail_rate_sbs": float(sbs_tail.mean()),
            "cap_rate_as": float(as_cap.mean()),
            "cap_rate_sbs": float(sbs_cap.mean()),
        })

    out = pd.DataFrame(rows)
    out["fgroup"] = pd.Categorical(out["fgroup"], categories=FGROUP_ORDER, ordered=True)
    out["Dim"] = pd.Categorical(out["Dim"], categories=BASE_DIMS_ORDER, ordered=True)
    return out.sort_values(["fgroup", "Dim"]).reset_index(drop=True)

def worst_cells(summ: pd.DataFrame, by="d_q", topk=3) -> pd.DataFrame:
    return summ.sort_values(by=by, ascending=False).head(topk).reset_index(drop=True)

def selection_histograms(df: pd.DataFrame, protocol: str, fgroup: str, dim: int, max_algos=12):
    g = df[(df["fgroup"] == fgroup) & (df["Dim"] == dim)].copy()
    if g.empty:
        print("Empty cell:", fgroup, dim)
        return

    def freq(s):
        return s.value_counts()

    overall = freq(g["chosen_algo"])
    tail = freq(g.loc[g["as_tail"], "chosen_algo"])
    cap = freq(g.loc[g["as_cap"], "chosen_algo"])

    algos = overall.index.tolist()
    for a in tail.index:
        if a not in algos: algos.append(a)
    for a in cap.index:
        if a not in algos: algos.append(a)
    algos = algos[:max_algos]

    def to_prop(counts):
        v = np.array([counts.get(a, 0) for a in algos], dtype=float)
        s = v.sum()
        return v / s if s > 0 else v

    p_all = to_prop(overall)
    p_tail = to_prop(tail)
    p_cap = to_prop(cap)

    x = np.arange(len(algos))
    w = 0.28

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(x - w, p_all, width=w, label="All")
    ax.bar(x, p_tail, width=w, label=f"Tail (relERT > {TAIL_T:g})")
    ax.bar(x + w, p_cap, width=w, label=f"Cap (relERT ≥ {CAP_T:g})")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion")
    ax.set_title(f"{protocol}: selections @ {fgroup}, d={dim}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

def margin_vs_failure(df: pd.DataFrame, protocol: str, n_bins=10):
    m = df["pred_margin"].to_numpy(dtype=float)
    if np.allclose(m, m[0]):
        print(f"{protocol}: margin has no variation; skipping.")
        return

    # quantile bins
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(m, qs))
    if len(edges) < 3:
        print(f"{protocol}: too few unique bin edges; skipping.")
        return

    bin_id = np.digitize(m, edges[1:-1], right=True)

    xs, tail_r, cap_r, cnt = [], [], [], []
    for b in range(bin_id.min(), bin_id.max() + 1):
        idx = (bin_id == b)
        if not np.any(idx):
            continue
        xs.append(float(np.median(m[idx])))
        tail_r.append(float(df.loc[idx, "as_tail"].mean()))
        cap_r.append(float(df.loc[idx, "as_cap"].mean()))
        cnt.append(int(idx.sum()))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(xs, tail_r, marker="o", label=f"Tail rate (>{TAIL_T:g})")
    ax.plot(xs, cap_r, marker="o", label=f"Cap rate (≥{CAP_T:g})")
    ax.set_xlabel("Prediction margin (median per bin)")
    ax.set_ylabel("Failure rate")
    ax.set_title(f"{protocol}: failure vs prediction margin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    return pd.DataFrame({"margin_median": xs, "tail_rate": tail_r, "cap_rate": cap_r, "count": cnt}).sort_values("margin_median")
