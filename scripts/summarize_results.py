#!/usr/bin/env python3
"""Standalone summary tool for medical image generation result CSV files.

The script automatically detects common metric, uncertainty and calibration
columns and writes a compact report plus machine-readable tables.

Examples
--------
python summarize_results.py results.csv
python summarize_results.py results.csv --out-dir summary_results --plots
python summarize_results.py calibration_bins.csv --group-by Type
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COMMON_ID_COLUMNS = {
    "sample", "sample_id", "case", "case_id", "subject", "subject_id",
    "filename", "file", "path", "type", "bin", "split", "fold"
}

ERROR_CANDIDATES = [
    "mae", "mse", "rmse", "error", "err_mean", "absolute_error",
    "mean_absolute_error", "l1", "nll"
]

UNCERTAINTY_CANDIDATES = [
    "u_mean", "unc_mean", "uncertainty", "uncertainty_mean", "variance_mean",
    "var_mean", "u_p95", "u_p99", "u_top1_mean", "u_top5_mean"
]

QUALITY_CANDIDATES = ["psnr", "ssim", "ms_ssim", "lpips"]


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {_norm(c): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for original in columns:
        normalized = _norm(original)
        for candidate in candidates:
            if candidate in normalized:
                return original
    return None


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0 and _norm(col) not in COMMON_ID_COLUMNS:
            cols.append(col)
    return cols


def _safe_sem(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _summary_table(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        sem = _safe_sem(values)
        rows.append({
            "metric": col,
            "count": int(values.size),
            "missing": int(len(df) - values.size),
            "mean": mean,
            "std": std,
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ci95_low": float(mean - 1.96 * sem) if np.isfinite(sem) else float("nan"),
            "ci95_high": float(mean + 1.96 * sem) if np.isfinite(sem) else float("nan"),
        })
    return pd.DataFrame(rows)


def _correlation_table(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame(columns=["x", "y", "pearson", "spearman", "n"])
    numeric = df[list(numeric_cols)].apply(pd.to_numeric, errors="coerce")
    rows = []
    for i, x in enumerate(numeric_cols):
        for y in numeric_cols[i + 1:]:
            pair = numeric[[x, y]].dropna()
            if len(pair) < 3:
                continue
            rows.append({
                "x": x,
                "y": y,
                "pearson": float(pair[x].corr(pair[y], method="pearson")),
                "spearman": float(pair[x].corr(pair[y], method="spearman")),
                "n": int(len(pair)),
            })
    return pd.DataFrame(rows)


def _risk_coverage(df: pd.DataFrame, uncertainty_col: str, error_col: str) -> pd.DataFrame:
    pair = df[[uncertainty_col, error_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        return pd.DataFrame(columns=["coverage", "risk", "retained", "threshold"])
    pair = pair.sort_values(uncertainty_col, ascending=True).reset_index(drop=True)
    n = len(pair)
    coverages = np.linspace(0.05, 1.0, 20)
    rows = []
    for coverage in coverages:
        retained = max(1, int(math.ceil(n * coverage)))
        subset = pair.iloc[:retained]
        rows.append({
            "coverage": float(retained / n),
            "risk": float(subset[error_col].mean()),
            "retained": int(retained),
            "threshold": float(subset[uncertainty_col].max()),
        })
    return pd.DataFrame(rows)


def _calibration_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    required = {
        "bin": _find_column(df.columns, ["bin"]),
        "unc": _find_column(df.columns, ["unc_mean", "uncertainty_mean", "u_mean"]),
        "err": _find_column(df.columns, ["err_mean", "error_mean", "mae", "mse"]),
        "count": _find_column(df.columns, ["count", "n"]),
    }
    if not all(required.values()):
        return None
    type_col = _find_column(df.columns, ["type"])
    work = df.copy()
    for key in ("bin", "unc", "err", "count"):
        work[required[key]] = pd.to_numeric(work[required[key]], errors="coerce")
    work = work.dropna(subset=[required["bin"], required["unc"], required["err"]])
    group_cols = [type_col] if type_col else []
    rows = []
    groups = work.groupby(group_cols, dropna=False) if group_cols else [("all", work)]
    for group_name, group in groups:
        weights = group[required["count"]].fillna(1.0).clip(lower=0).to_numpy(float)
        unc = group[required["unc"]].to_numpy(float)
        err = group[required["err"]].to_numpy(float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weighted_gap = float(np.average(np.abs(unc - err), weights=weights))
        rows.append({
            "type": str(group_name),
            "bins": int(len(group)),
            "total_count": float(weights.sum()),
            "weighted_abs_calibration_gap": weighted_gap,
            "unc_error_pearson": float(pd.Series(unc).corr(pd.Series(err), method="pearson")) if len(group) > 1 else float("nan"),
            "unc_error_spearman": float(pd.Series(unc).corr(pd.Series(err), method="spearman")) if len(group) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def _format_number(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4f}"


def _write_markdown(
    path: Path,
    source: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    risk: Optional[pd.DataFrame],
    calibration: Optional[pd.DataFrame],
    uncertainty_col: Optional[str],
    error_col: Optional[str],
) -> None:
    lines = [
        "# Results summary",
        "",
        f"- Source: `{source}`",
        f"- Rows: **{len(df)}**",
        f"- Columns: **{len(df.columns)}**",
        f"- Duplicate rows: **{int(df.duplicated().sum())}**",
        "",
        "## Numeric metrics",
        "",
    ]
    if summary.empty:
        lines.append("No numeric metric columns were detected.")
    else:
        lines.append("| Metric | N | Mean ± SD | Median [Q1, Q3] | Min–Max | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['metric']} | {int(row['count'])} | "
                f"{_format_number(row['mean'])} ± {_format_number(row['std'])} | "
                f"{_format_number(row['median'])} [{_format_number(row['q1'])}, {_format_number(row['q3'])}] | "
                f"{_format_number(row['min'])}–{_format_number(row['max'])} | "
                f"[{_format_number(row['ci95_low'])}, {_format_number(row['ci95_high'])}] |"
            )

    if uncertainty_col and error_col:
        pair = df[[uncertainty_col, error_col]].apply(pd.to_numeric, errors="coerce").dropna()
        lines += [
            "",
            "## Uncertainty–error association",
            "",
            f"- Uncertainty column: `{uncertainty_col}`",
            f"- Error column: `{error_col}`",
        ]
        if len(pair) >= 3:
            lines += [
                f"- Pearson correlation: **{_format_number(pair[uncertainty_col].corr(pair[error_col], method='pearson'))}**",
                f"- Spearman correlation: **{_format_number(pair[uncertainty_col].corr(pair[error_col], method='spearman'))}**",
            ]

    if calibration is not None and not calibration.empty:
        lines += ["", "## Calibration-bin summary", ""]
        lines.append("| Type | Bins | Total count | Weighted absolute gap | Pearson | Spearman |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in calibration.iterrows():
            lines.append(
                f"| {row['type']} | {int(row['bins'])} | {_format_number(row['total_count'])} | "
                f"{_format_number(row['weighted_abs_calibration_gap'])} | "
                f"{_format_number(row['unc_error_pearson'])} | {_format_number(row['unc_error_spearman'])} |"
            )

    if risk is not None and not risk.empty:
        full_risk = float(risk.iloc[-1]["risk"])
        low_cov = risk.iloc[(risk["coverage"] - 0.5).abs().argmin()]
        lines += [
            "",
            "## Selective prediction",
            "",
            f"- Risk at full coverage: **{_format_number(full_risk)}**",
            f"- Risk near 50% coverage: **{_format_number(low_cov['risk'])}**",
            f"- Relative risk reduction near 50% coverage: **{_format_number((full_risk - float(low_cov['risk'])) / full_risk if full_risk != 0 else np.nan)}**",
        ]

    if not correlations.empty:
        strongest = correlations.assign(abs_spearman=correlations["spearman"].abs()).sort_values("abs_spearman", ascending=False).head(10)
        lines += ["", "## Strongest pairwise correlations", ""]
        lines.append("| X | Y | Pearson | Spearman | N |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in strongest.iterrows():
            lines.append(
                f"| {row['x']} | {row['y']} | {_format_number(row['pearson'])} | "
                f"{_format_number(row['spearman'])} | {int(row['n'])} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plots(
    out_dir: Path,
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    uncertainty_col: Optional[str],
    error_col: Optional[str],
    risk: Optional[pd.DataFrame],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; plots were skipped.", file=sys.stderr)
        return

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(values.to_numpy(), bins=min(30, max(5, int(math.sqrt(len(values))))))
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{_norm(col)}.png", dpi=160)
        plt.close()

    if uncertainty_col and error_col:
        pair = df[[uncertainty_col, error_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 2:
            plt.figure(figsize=(6, 5))
            plt.scatter(pair[uncertainty_col], pair[error_col], s=16, alpha=0.65)
            plt.xlabel(uncertainty_col)
            plt.ylabel(error_col)
            plt.title("Uncertainty vs error")
            plt.tight_layout()
            plt.savefig(out_dir / "uncertainty_vs_error.png", dpi=160)
            plt.close()

    if risk is not None and not risk.empty:
        plt.figure(figsize=(6, 5))
        plt.plot(risk["coverage"], risk["risk"], marker="o")
        plt.xlabel("Coverage")
        plt.ylabel("Risk")
        plt.title("Risk–coverage curve")
        plt.tight_layout()
        plt.savefig(out_dir / "risk_coverage.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize metric, uncertainty and calibration analyses from a CSV file."
    )
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--sep", default=None, help="CSV separator; auto-detected by default")
    parser.add_argument("--encoding", default="utf-8", help="Input encoding")
    parser.add_argument("--uncertainty-col", default=None, help="Override uncertainty column")
    parser.add_argument("--error-col", default=None, help="Override error/risk column")
    parser.add_argument("--group-by", default=None, help="Optional categorical column for grouped summaries")
    parser.add_argument("--plots", action="store_true", help="Generate PNG plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.csv.expanduser().resolve()
    if not source.exists():
        print(f"Error: input file does not exist: {source}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or source.with_name(f"{source.stem}_summary")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.sep is None:
            df = pd.read_csv(source, sep=None, engine="python", encoding=args.encoding)
        else:
            df = pd.read_csv(source, sep=args.sep, encoding=args.encoding)
    except Exception as exc:
        print(f"Error while reading CSV: {exc}", file=sys.stderr)
        return 3

    if df.empty:
        print("Error: the CSV contains no rows.", file=sys.stderr)
        return 4

    numeric_cols = _numeric_columns(df)
    summary = _summary_table(df, numeric_cols)
    correlations = _correlation_table(df, numeric_cols)

    uncertainty_col = args.uncertainty_col or _find_column(df.columns, UNCERTAINTY_CANDIDATES)
    error_col = args.error_col or _find_column(df.columns, ERROR_CANDIDATES)

    if uncertainty_col and uncertainty_col not in df.columns:
        print(f"Error: uncertainty column not found: {uncertainty_col}", file=sys.stderr)
        return 5
    if error_col and error_col not in df.columns:
        print(f"Error: error column not found: {error_col}", file=sys.stderr)
        return 6

    risk = _risk_coverage(df, uncertainty_col, error_col) if uncertainty_col and error_col else None
    calibration = _calibration_summary(df)

    summary.to_csv(out_dir / "numeric_summary.csv", index=False)
    correlations.to_csv(out_dir / "correlations.csv", index=False)
    if risk is not None:
        risk.to_csv(out_dir / "risk_coverage.csv", index=False)
    if calibration is not None:
        calibration.to_csv(out_dir / "calibration_summary.csv", index=False)

    if args.group_by:
        if args.group_by not in df.columns:
            print(f"Error: group-by column not found: {args.group_by}", file=sys.stderr)
            return 7
        grouped_rows = []
        for group_value, group_df in df.groupby(args.group_by, dropna=False):
            group_summary = _summary_table(group_df, numeric_cols)
            group_summary.insert(0, args.group_by, group_value)
            grouped_rows.append(group_summary)
        if grouped_rows:
            pd.concat(grouped_rows, ignore_index=True).to_csv(out_dir / "grouped_numeric_summary.csv", index=False)

    payload = {
        "source": str(source),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(numeric_cols),
        "uncertainty_column": uncertainty_col,
        "error_column": error_col,
        "numeric_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "calibration_summary": None if calibration is None else calibration.replace({np.nan: None}).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _write_markdown(
        out_dir / "summary.md",
        source,
        df,
        summary,
        correlations,
        risk,
        calibration,
        uncertainty_col,
        error_col,
    )

    if args.plots:
        _make_plots(out_dir, df, numeric_cols, uncertainty_col, error_col, risk)

    print(f"Summary written to: {out_dir}")
    print(f"Main report: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
