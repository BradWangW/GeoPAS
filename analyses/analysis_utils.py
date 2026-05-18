from __future__ import annotations

import csv
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ModuleNotFoundError:
    mpl = None
    plt = None
    LinearSegmentedColormap = None


ALGORITHM_COLUMNS = [
    "BrentSTEPqi_Posik",
    "BrentSTEPrr_Posik",
    "CMA-CSA_Atamna",
    "HCMA_loshchilov_noiseless",
    "HMLSL_pal_noiseless",
    "IPOP400D_auger_noiseless",
    "MCS_huyer_noiseless",
    "M_LSL_pal_noiseless",
    "OQNLP_pal_noiseless",
    "SMAC-BBOB_hutter_noiseless",
    "fmincon_pal_noiseless",
    "fminunc_pal_noiseless",
]

PROBLEM_GROUP_ORDER = ["f1-f5", "f6-f9", "f10-f14", "f15-f19", "f20-f24", "all"]
DIM_ORDER = [2, 3, 5, 10, "all"]

RESULT_SPLITS = ("LPO", "LIO", "RANDOM")
PRIMARY_METRICS = ("Mean", "Median", "P90")
OPTIONAL_METRICS = ("Log_Mean", "Log_Median", "Log_P90")
ALL_SECTIONED_METRICS = PRIMARY_METRICS + OPTIONAL_METRICS
SUMMARY_SECTIONS = ("AS", "VBS", "SBS", "Gap_Closure")
SINGLE_TABLE_METRICS = {
    "Accuracies": "accuracy",
    "F1": "f1",
}
STOP_LABELS = set(PRIMARY_METRICS) | set(SUMMARY_SECTIONS) | set(SINGLE_TABLE_METRICS) | {
    "Pick_Rate",
    "VBS_Pick_Rate",
}

RUN_PARAMETER_COLUMNS = (
    "target_scale",
    "head_2_target_scale",
    "prior_scale",
    "sigmoid_log_s",
    "tail_scale",
    "lam_prior",
    "dual_head",
    "head_2_loss_weight",
    "head_2_score_weight",
    "res",
    "k_views",
)

NUMBER_PATTERN = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
PARAM_PATTERN = re.compile(
    rf"^scale(?P<target_scale>.+?)_head2scale(?P<head_2_target_scale>.+?)"
    rf"_sigmoidlogs(?P<sigmoid_log_s>{NUMBER_PATTERN})_dual(?P<dual_head>\d+)"
    rf"_head2lw(?P<head_2_loss_weight>{NUMBER_PATTERN})_head2sw(?P<head_2_score_weight>{NUMBER_PATTERN})"
    rf"_priorscale(?P<prior_scale>.+?)_lamprior(?P<lam_prior>{NUMBER_PATTERN})"
    rf"_tailscale(?P<tail_scale>{NUMBER_PATTERN})$"
)
RUN_DIR_RE = re.compile(
    rf"scale(?P<target_scale>[A-Za-z0-9_]+)"
    rf"_head2scale(?P<head_2_target_scale>[A-Za-z0-9_]+)"
    rf"_sigmoidlogs(?P<sigmoid_log_s>{NUMBER_PATTERN})"
    rf"_dual(?P<dual_head>\d+)"
    rf"_head2lw(?P<head_2_loss_weight>{NUMBER_PATTERN})"
    rf"_head2sw(?P<head_2_score_weight>{NUMBER_PATTERN})"
    rf"_priorscale(?P<prior_scale>[A-Za-z0-9_]+)"
    rf"_lamprior(?P<lam_prior>{NUMBER_PATTERN})"
    rf"_tailscale(?P<tail_scale>{NUMBER_PATTERN})"
    rf"_res_(?P<res>\d+)"
    rf"_k_views_(?P<k_views>\d+)"
)
RUN_DIR_SUFFIX_RE = re.compile(r"^(?P<parameter_set>.+)_res_(?P<res>\d+)_k_views_(?P<k_views>\d+)$")

PAPER_BG = "#f9f8f5"
AXIS_BG = "#ffffff"
GRID_COLOR = "#d8cec0"
TEXT_COLOR = "#2d2622"
BOX_EDGE_COLOR = "#cbbfab"
MISSING_COLOR = "#efe7dc"

PORTFOLIO_CMAP = (
    LinearSegmentedColormap.from_list(
        "as_bbo_portfolio",
        ["#143d4a", "#336c7a", "#7aa7a2", "#d7d3b1", "#f3eadf"],
    )
    if LinearSegmentedColormap is not None
    else None
)
DARKER_PORTFOLIO_CMAP = (
    LinearSegmentedColormap.from_list(
        "as_bbo_portfolio_darker",
        ["#0f2e36", "#2a5468", "#5c8b8f", "#b9b37e", "#d9d5c4"],
    )
    if LinearSegmentedColormap is not None
    else None
)
HEATMAP_CMAP = (
    LinearSegmentedColormap.from_list(
        "as_bbo_heatmap",
        ["#143d4a", "#336c7a", "#7aa7a2", "#d7d3b1", "#f3eadf"],
    )
    if LinearSegmentedColormap is not None
    else None
)
if HEATMAP_CMAP is not None:
    HEATMAP_CMAP.set_bad(MISSING_COLOR)

PARAM_LABELS = {
    "target_scale": "Target scale",
    "head_2_target_scale": "Head-2 target scale",
    "prior_scale": "Prior scale",
    "sigmoid_log_s": "Sigmoid-log s",
    "tail_scale": "Tail scale",
    "lam_prior": "Prior weight",
    "dual_head": "Dual head",
    "head_2_loss_weight": "Head-2 loss weight",
    "head_2_score_weight": "Head-2 score weight",
    "res": "Resolution",
    "k_views": "K views",
}


@dataclass(frozen=True)
class GeopasPaths:
    workspace_root: Path
    project_root: Path
    results_root: Path
    bbob_results_root: Path
    relert_path: Path
    analysis_output_root: Path


class ParseError(RuntimeError):
    pass


def find_workspace_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "train_parallel.py").exists() and (candidate / "functions").exists():
            return candidate
    raise FileNotFoundError("Could not locate the GeoPAS workspace root from the current working directory.")


def resolve_geopas_paths(start: Path | None = None) -> GeopasPaths:
    workspace_root = find_workspace_root(start)
    project_root = Path(
        os.environ.get("PROJECT_ROOT", os.environ.get("GEOPAS_PROJECT_ROOT", str(workspace_root.parent)))
    ).resolve()
    results_root = Path(
        os.environ.get("RESULTS_ROOT", str(project_root / "results" / "bbob_by_deepela" / "results"))
    ).resolve()

    relert_candidates = [
        workspace_root / "data_generation" / "performances" / "relert.csv",
        project_root / "data" / "bbob_by_deepela" / "relert.csv",
    ]
    relert_path = next((candidate for candidate in relert_candidates if candidate.exists()), relert_candidates[0])

    return GeopasPaths(
        workspace_root=workspace_root,
        project_root=project_root,
        results_root=results_root,
        bbob_results_root=results_root / "bbob",
        relert_path=relert_path,
        analysis_output_root=workspace_root / "analysis_outputs",
    )


def _require_matplotlib() -> tuple[object, object]:
    if mpl is None or plt is None:
        raise ModuleNotFoundError("matplotlib is required for the plotting helpers in analyses.analysis_utils")
    return mpl, plt


def apply_paper_theme(*, figure_dpi: int = 140, savefig_dpi: int = 240, title_size: int = 18) -> None:
    mpl_module, _ = _require_matplotlib()
    mpl_module.rcParams.update(
        {
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": AXIS_BG,
            "axes.edgecolor": AXIS_BG,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "font.family": "DejaVu Serif",
            "axes.titlesize": title_size,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "savefig.facecolor": PAPER_BG,
            "savefig.bbox": "tight",
        }
    )


def scale_relert_values(
    frame: pd.DataFrame,
    *,
    algorithm_columns: Sequence[str] = ALGORITHM_COLUMNS,
    use_log: bool = False,
) -> pd.DataFrame:
    scaled = frame.copy()
    values = np.log(scaled.loc[:, list(algorithm_columns)]) if use_log else scaled.loc[:, list(algorithm_columns)]
    min_value = values.min().min()
    max_value = values.max().max()
    span = max_value - min_value
    scaled.loc[:, list(algorithm_columns)] = 0.0 if span == 0 else (values - min_value) / span
    return scaled


def parse_parameter_set(parameter_set: str) -> dict[str, str]:
    match = PARAM_PATTERN.fullmatch(parameter_set)
    if match is None:
        raise ParseError(f"Could not parse parameter_set={parameter_set!r}")
    return match.groupdict()


def build_run_dir_name(parameter_set: str, *, res: int, k_views: int) -> str:
    return f"{parameter_set}_res_{int(res)}_k_views_{int(k_views)}"


def build_protocol_paths(
    *,
    results_root: Path,
    protocol: str,
    parameter_set: str,
    res: int,
    k_views: int,
) -> dict[str, Path]:
    tokens = parse_parameter_set(parameter_set)
    base = Path(results_root) / protocol.lower() / build_run_dir_name(parameter_set, res=res, k_views=k_views)
    stem = (
        f"priorscale_{tokens['prior_scale']}_sigmoidlogs_{tokens['sigmoid_log_s']}_"
        f"tailscale_{tokens['tail_scale']}_head2lossweight_{tokens['head_2_loss_weight']}_"
        f"head2scoreweight_{tokens['head_2_score_weight']}_"
        f"head2targetscale_{tokens['head_2_target_scale']}_lamprior_{tokens['lam_prior']}_"
        f"res_{int(res)}_k_views_{int(k_views)}"
    )
    return {
        "base": base,
        "pred": base / f"preds_{stem}.csv.gz",
        "result": base / f"res_{stem}.csv",
    }


def fgroup_from_problem(problem: str | int) -> str:
    fid = int(str(problem).lower().lstrip("f"))
    if 1 <= fid <= 5:
        return "f1-f5"
    if 6 <= fid <= 9:
        return "f6-f9"
    if 10 <= fid <= 14:
        return "f10-f14"
    if 15 <= fid <= 19:
        return "f15-f19"
    if 20 <= fid <= 24:
        return "f20-f24"
    raise ParseError(f"Unsupported problem label: {problem!r}")


def ensure_algorithm_columns(
    df: pd.DataFrame,
    meta_cols: Sequence[str],
    file_label: str,
    *,
    expected_columns: Sequence[str] = ALGORITHM_COLUMNS,
) -> None:
    actual = [col for col in df.columns if col not in meta_cols]
    if actual != list(expected_columns):
        raise ParseError(
            f"{file_label} algorithm columns do not match the expected ordering.\n"
            f"Expected: {list(expected_columns)}\nActual:   {actual}"
        )


def _skip_blank_lines(lines: Sequence[str], start_idx: int) -> int:
    index = int(start_idx)
    while index < len(lines) and not str(lines[index]).strip():
        index += 1
    return index


def _parse_csv_row(line: str) -> list[str]:
    return next(csv.reader([line]))


def _read_table(lines: Sequence[str], start_idx: int, *, stop_labels: set[str]) -> tuple[pd.DataFrame, int]:
    index = _skip_blank_lines(lines, start_idx)
    rows: list[str] = []
    while index < len(lines):
        token = str(lines[index]).strip()
        if not token or token in stop_labels:
            break
        rows.append(str(lines[index]))
        index += 1

    if not rows:
        return pd.DataFrame(), index
    return pd.read_csv(io.StringIO("\n".join(rows))), index


def _extract_table_block(lines: Sequence[str], stat: str, substat: str) -> pd.DataFrame:
    start = None
    for index in range(len(lines) - 1):
        if str(lines[index]).strip() == stat and str(lines[index + 1]).strip() == substat:
            start = index + 2
            break
    if start is None:
        raise ParseError(f"Could not find section {stat!r} -> {substat!r}")

    table, _ = _read_table(lines, start, stop_labels=STOP_LABELS)
    if table.empty or table.shape[1] < 2:
        raise ParseError(f"Parsed table for {stat!r} -> {substat!r} looks incomplete")
    return table.set_index(table.columns[0])


def read_as_tables(csv_path: Path, *, metrics: Sequence[str] = PRIMARY_METRICS, substat: str = "AS") -> dict[str, pd.DataFrame]:
    lines = Path(csv_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {metric: _extract_table_block(lines, metric, substat) for metric in metrics}


def parse_result_csv_tables(result_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
    lines = Path(result_path).read_text(encoding="utf-8", errors="replace").splitlines()
    parsed: dict[str, dict[str, pd.DataFrame]] = {}
    index = 0

    while index < len(lines):
        index = _skip_blank_lines(lines, index)
        if index >= len(lines):
            break

        label = lines[index].strip()
        if label not in PRIMARY_METRICS:
            index += 1
            continue

        index += 1
        blocks: dict[str, pd.DataFrame] = {}
        for section in SUMMARY_SECTIONS:
            index = _skip_blank_lines(lines, index)
            if index >= len(lines) or lines[index].strip() != section:
                raise ParseError(f"Expected section {section!r} inside {label}, got {lines[index].strip()!r}")
            index += 1
            blocks[section], index = _read_table(lines, index, stop_labels=STOP_LABELS)
        parsed[label] = blocks

    return parsed


def _maybe_shorten_section_name(section: str, *, shorten: bool) -> str:
    if not shorten:
        return section
    return section.replace("_priorscalesigmoid_log", "").replace("_tailscale1.0", "")


def _extract_parameter_set(path_token: str, *, shorten: bool = False) -> str:
    match = RUN_DIR_SUFFIX_RE.fullmatch(path_token)
    parameter_set = match.group("parameter_set") if match else path_token
    return _maybe_shorten_section_name(parameter_set, shorten=shorten)


def iter_split_result_csv_paths(
    split: str,
    *,
    results_root: Path,
    filename_glob: str = "res_priorscale_*.csv",
) -> list[Path]:
    split_root = Path(results_root) / split.lower()
    if not split_root.is_dir():
        return []
    return sorted(split_root.glob(f"*/{filename_glob}"))


def build_section_name(
    csv_path: Path,
    *,
    split: str,
    grid_root: Path,
    shorten: bool = False,
) -> str:
    parts = Path(csv_path).relative_to(grid_root).parts
    if len(parts) != 3 or parts[0].lower() != split.lower():
        raise ParseError(f"Unexpected aggregated path layout for {csv_path}")
    return " -- ".join([_extract_parameter_set(parts[1], shorten=shorten), Path(csv_path).stem])


def build_sectioned_csv_for_split(
    split: str,
    *,
    results_root: Path,
    shorten_section_names: bool = False,
    filename_glob: str = "res_priorscale_*.csv",
) -> Optional[Path]:
    csv_paths = iter_split_result_csv_paths(split, results_root=results_root / "bbob", filename_glob=filename_glob)
    if not csv_paths:
        return None

    out_path = Path(results_root) / f"AS_mean_median_p90__{split}__ALL_RUNS.csv"
    grid_root = Path(results_root) / "bbob"

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for csv_path in csv_paths:
            try:
                tables = read_as_tables(csv_path)
            except Exception:
                continue

            handle.write(f"{build_section_name(csv_path, split=split, grid_root=grid_root, shorten=shorten_section_names)}\n")
            for metric in PRIMARY_METRICS:
                handle.write(f"{metric}\n")
                tables[metric].to_csv(handle)
            handle.write("\n")

    return out_path


def build_all_available_splits(
    splits: Sequence[str],
    *,
    results_root: Path,
    shorten_section_names: bool = False,
    filename_glob: str = "res_priorscale_*.csv",
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for split in splits:
        out_path = build_sectioned_csv_for_split(
            split,
            results_root=results_root,
            shorten_section_names=shorten_section_names,
            filename_glob=filename_glob,
        )
        if out_path is not None:
            outputs[split] = out_path
    return outputs


def _consume_metric_block(lines: Sequence[str], index: int, metric: str, path: Path) -> tuple[list[str], int]:
    if index >= len(lines) or str(lines[index]).strip() != metric:
        raise ParseError(f"{path}: expected {metric!r} at line {index + 1}")

    index += 1
    rows: list[str] = []
    while index < len(lines):
        token = str(lines[index]).strip()
        if not token or token in ALL_SECTIONED_METRICS:
            break
        rows.append(str(lines[index]))
        index += 1
    return rows, index


def parse_sectioned_csv(
    path: Path,
    *,
    primary_metrics: Sequence[str] = PRIMARY_METRICS,
    optional_metrics: Sequence[str] = OPTIONAL_METRICS,
) -> dict[str, dict[str, list[str]]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    parsed: dict[str, dict[str, list[str]]] = {}
    index = 0

    while True:
        index = _skip_blank_lines(lines, index)
        if index >= len(lines):
            break

        section = lines[index].strip()
        index += 1
        metrics: dict[str, list[str]] = {}

        for metric in primary_metrics:
            index = _skip_blank_lines(lines, index)
            metrics[metric], index = _consume_metric_block(lines, index, metric, Path(path))

        for metric in optional_metrics:
            index = _skip_blank_lines(lines, index)
            if index < len(lines) and lines[index].strip() == metric:
                metrics[metric], index = _consume_metric_block(lines, index, metric, Path(path))
            else:
                metrics[metric] = []

        parsed[section] = metrics

    return parsed


def _last_metric_value(rows: Sequence[str]) -> float:
    if not rows:
        return float("inf")
    return float(str(rows[-1]).rsplit(",", 1)[-1])


def merge_sectioned_split_csvs(
    files: Mapping[str, Path],
    *,
    out_path: Path,
    sort_by_split: str = "LPO",
) -> Optional[Path]:
    available_files = {split: Path(path) for split, path in files.items() if Path(path).exists()}
    if not available_files:
        return None

    parsed = {split: parse_sectioned_csv(path) for split, path in available_files.items()}
    section_order = list(parsed[next(iter(parsed))].keys())

    if sort_by_split in parsed:
        original_index = {section: index for index, section in enumerate(section_order)}
        section_order.sort(
            key=lambda section: (
                section not in parsed[sort_by_split],
                _last_metric_value(parsed[sort_by_split].get(section, {}).get("Median", [])),
                original_index[section],
            )
        )

    with Path(out_path).open("w", encoding="utf-8", newline="\n") as handle:
        for section in section_order:
            handle.write(section + "\n")
            for split, split_data in available_files.items():
                split_metrics = parsed[split].get(section)
                if split_metrics is None:
                    continue
                handle.write(split + "\n")
                for metric in ALL_SECTIONED_METRICS:
                    rows = split_metrics.get(metric, [])
                    if not rows:
                        continue
                    handle.write(metric + "\n")
                    handle.write("\n".join(rows) + "\n")
            handle.write("\n")

    return Path(out_path)


def _parse_all_all_value(lines: Sequence[str], start_idx: int, label: str) -> tuple[float, int]:
    index = _skip_blank_lines(lines, start_idx)
    if index >= len(lines) or not str(lines[index]).lstrip().startswith("Problem Group"):
        got = str(lines[index]).rstrip() if index < len(lines) else "<EOF>"
        raise ParseError(f"Expected 'Problem Group,...' after {label}, got: {got}")
    index += 1

    all_row: Optional[list[str]] = None
    while index < len(lines):
        token = str(lines[index]).strip()
        if not token or token in STOP_LABELS:
            break
        row = [field.strip() for field in _parse_csv_row(str(lines[index]))]
        if row and row[0].lower() == "all":
            all_row = row
        index += 1

    if all_row is None:
        raise ParseError(f"Did not find an 'all' row for {label}")

    try:
        return float(all_row[-1]), index
    except ValueError as exc:
        raise ParseError(f"Could not parse {label} all/all value {all_row[-1]!r}") from exc


def parse_run_dir_name(run_dir_name: str) -> dict[str, object]:
    match = RUN_DIR_RE.fullmatch(run_dir_name)
    if match is None:
        raise ParseError(f"Unrecognised run directory name: {run_dir_name}")
    groups = match.groupdict()
    return {
        "target_scale": groups["target_scale"],
        "head_2_target_scale": groups["head_2_target_scale"],
        "prior_scale": groups["prior_scale"],
        "sigmoid_log_s": float(groups["sigmoid_log_s"]),
        "tail_scale": float(groups["tail_scale"]),
        "lam_prior": float(groups["lam_prior"]),
        "dual_head": int(groups["dual_head"]),
        "head_2_loss_weight": float(groups["head_2_loss_weight"]),
        "head_2_score_weight": float(groups["head_2_score_weight"]),
        "res": int(groups["res"]),
        "k_views": int(groups["k_views"]),
    }


def parse_result_summary_csv(csv_path: str | Path) -> dict[str, float]:
    lines = Path(csv_path).read_text(encoding="utf-8", errors="replace").splitlines()
    metrics: dict[str, float] = {}
    index = 0

    while index < len(lines):
        index = _skip_blank_lines(lines, index)
        if index >= len(lines):
            break

        label = lines[index].strip()
        if label in PRIMARY_METRICS:
            metric_key = label.lower()
            index += 1
            section_values: dict[str, float] = {}
            for section in SUMMARY_SECTIONS:
                index = _skip_blank_lines(lines, index)
                if index >= len(lines) or lines[index].strip() != section:
                    got = lines[index].strip() if index < len(lines) else "<EOF>"
                    raise ParseError(f"Expected section {section!r} inside {label}, got {got!r}")
                index += 1
                section_values[section], index = _parse_all_all_value(lines, index, f"{label}/{section}")

            metrics[metric_key] = section_values["AS"]
            metrics[f"sbs_{metric_key}"] = section_values["SBS"]
            metrics[f"vbs_{metric_key}"] = section_values["VBS"]
            metrics[f"gap_closure_{metric_key}"] = section_values["Gap_Closure"]
            continue

        if label in SINGLE_TABLE_METRICS:
            metric_key = SINGLE_TABLE_METRICS[label]
            index += 1
            metrics[metric_key], index = _parse_all_all_value(lines, index, label)
            continue

        if label in {"Pick_Rate", "VBS_Pick_Rate"}:
            break

        index += 1

    return metrics


def matches_filter(actual: object, expected: object) -> bool:
    if expected is None:
        return True
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(matches_filter(actual, option) for option in expected)
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
    return actual == expected


def filter_dataframe_by_user_filters(df: pd.DataFrame, filters: Mapping[str, object]) -> pd.DataFrame:
    filtered = df.copy()
    for key, expected in filters.items():
        if key not in filtered.columns or expected is None:
            continue
        filtered = filtered[filtered[key].map(lambda actual: matches_filter(actual, expected))]
    return filtered.reset_index(drop=True)


def find_result_summary_csv(run_dir: Path, seed: Optional[int] = None) -> Optional[Path]:
    base_dir = Path(run_dir) if seed is None else Path(run_dir) / f"seed{int(seed)}"
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob("res_*.csv"))
    return candidates[0] if len(candidates) == 1 else None


def list_available_parameter_slices(
    filters: Mapping[str, object],
    *,
    varying_cols: Sequence[str],
    results_root: Path,
) -> pd.DataFrame:
    protocol = str(filters["protocol"])
    protocol_dir = Path(results_root) / protocol
    if not protocol_dir.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for run_dir in sorted(protocol_dir.iterdir()):
        if not run_dir.is_dir() or find_result_summary_csv(run_dir, seed=filters.get("seed")) is None:
            continue
        try:
            rows.append(parse_run_dir_name(run_dir.name))
        except ParseError:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    slice_cols = [column for column in RUN_PARAMETER_COLUMNS if column not in set(varying_cols)]
    available = (
        df.groupby(slice_cols, dropna=False)
        .size()
        .rename("n_plot_parameter_pairs")
        .reset_index()
        .sort_values(slice_cols)
        .reset_index(drop=True)
    )
    return filter_dataframe_by_user_filters(available, filters)


def collect_parameter_grid_results(
    filters: Mapping[str, object],
    *,
    x_col: str,
    y_col: str,
    results_root: Path,
) -> pd.DataFrame:
    protocol = str(filters["protocol"])
    protocol_dir = Path(results_root) / protocol
    if not protocol_dir.exists():
        return pd.DataFrame()

    fixed_filter_keys = {
        key: value
        for key, value in filters.items()
        if key not in {"protocol", "seed", x_col, y_col}
    }

    rows: list[dict[str, object]] = []
    for run_dir in sorted(protocol_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_csv = find_result_summary_csv(run_dir, seed=filters.get("seed"))
        if summary_csv is None:
            continue
        try:
            params = parse_run_dir_name(run_dir.name)
        except ParseError:
            continue

        if any(key not in params or not matches_filter(params[key], expected) for key, expected in fixed_filter_keys.items()):
            continue

        rows.append({"protocol": protocol, "csv_path": str(summary_csv), **params, **parse_result_summary_csv(summary_csv)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values([y_col, x_col]).reset_index(drop=True)
    if df.duplicated(subset=[x_col, y_col], keep=False).any():
        return pd.DataFrame()
    df["budget"] = df["res"] ** 2 * df["k_views"]
    return df


def _pretty_name(name: str) -> str:
    return PARAM_LABELS.get(name, name.replace("_", " ").title())


def _format_axis_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(numeric)}" if numeric.is_integer() else f"{numeric:g}"


def _format_cell_value(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    order = int(np.floor(np.log10(magnitude)))
    decimals = min(3, max(0, 3 - order))
    rounded = round(value, decimals)
    if rounded == 0:
        return "0"
    if decimals == 0:
        return f"{rounded:.0f}"
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def _text_color_from_tile(im: mpl.image.AxesImage, value: float) -> str:
    normed = im.norm(value)
    if np.ma.is_masked(normed):
        return TEXT_COLOR

    normed_array = np.asarray(normed)
    if normed_array.size == 0 or not np.isfinite(normed_array).all():
        return TEXT_COLOR

    rgba = np.asarray(im.cmap(float(normed_array.reshape(-1)[0])))
    if rgba.ndim > 1:
        rgba = rgba.reshape(-1, rgba.shape[-1])[0]
    if rgba.size < 4:
        return TEXT_COLOR

    red, green, blue, _ = rgba[:4]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return AXIS_BG if luminance < 0.6 else TEXT_COLOR


def plot_parameter_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    *,
    x_col: str,
    y_col: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    cbar_range: Optional[tuple[float, float]] = None,
    cmap: Optional[mpl.colors.Colormap] = None,
    figsize: Optional[tuple[float, float]] = None,
    log_scale: bool = False,
) -> None:
    mpl_module, plt_module = _require_matplotlib()
    pivot = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    data = np.ma.masked_invalid(pivot.values.astype(float))

    auto_figsize = figsize or (0.68 * len(pivot.columns) + 1, 0.68 * len(pivot.index) + 1)
    fig, ax = plt_module.subplots(figsize=auto_figsize)

    norm = None
    if log_scale:
        finite_positive = pivot.values[np.isfinite(pivot.values) & (pivot.values > 0)]
        if finite_positive.size and not np.allclose(finite_positive.min(), finite_positive.max()):
            norm = mpl_module.colors.LogNorm(vmin=float(finite_positive.min()), vmax=float(finite_positive.max()))

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap or HEATMAP_CMAP,
        interpolation="nearest",
        norm=norm,
    )
    if cbar_range is not None:
        im.set_clim(*cbar_range)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_format_axis_value(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_format_axis_value(value) for value in pivot.index])
    ax.set_xlabel(x_label or _pretty_name(x_col))
    ax.set_ylabel(y_label or _pretty_name(y_col))

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y_index in range(pivot.shape[0]):
        for x_index in range(pivot.shape[1]):
            value = pivot.values[y_index, x_index]
            if np.isnan(value):
                continue
            ax.text(
                x_index,
                y_index,
                _format_cell_value(float(value)),
                ha="center",
                va="center",
                color=_text_color_from_tile(im, float(value)),
                fontsize=11,
                fontweight="semibold",
            )

    protocol_label = str(df["protocol"].iloc[0]).upper()
    ax.set_title(f"{title} ({protocol_label})", y=1.02, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if output_path is not None:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_scale = str(df["target_scale"].iloc[0]) if "target_scale" in df.columns else "grid"
        fig.savefig(output_dir / f"{protocol_label.lower()}_{target_scale}_{value_col}_{y_col}_vs_{x_col}_heatmap.svg")

    plt_module.show()