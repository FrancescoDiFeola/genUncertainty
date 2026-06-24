from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ["classe", "latent_dist", "PSNR", "SSIM", "MAE"]


def load_metrics(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=None, engine="python")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.copy()
    df["classe"] = df["classe"].astype(str)

    for col in ["latent_dist", "PSNR", "SSIM", "MAE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    if len(df) == 0:
        raise ValueError("No valid rows after removing missing/non-numeric values.")

    return df


def add_failure_labels(df):
    df = df.copy()

    psnr_thr = df["PSNR"].quantile(0.25)
    ssim_thr = df["SSIM"].quantile(0.25)
    mae_thr = df["MAE"].quantile(0.75)

    df["failure_psnr"] = df["PSNR"] <= psnr_thr
    df["failure_ssim"] = df["SSIM"] <= ssim_thr
    df["failure_mae"] = df["MAE"] >= mae_thr

    df["failure_any"] = (
        df["failure_psnr"] | df["failure_ssim"] | df["failure_mae"]
    )

    return df


def scenario_summary(df):
    summary = (
        df.groupby("classe")
        .agg(
            n=("classe", "size"),
            latent_dist_mean=("latent_dist", "mean"),
            latent_dist_std=("latent_dist", "std"),
            PSNR_mean=("PSNR", "mean"),
            PSNR_std=("PSNR", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            failure_rate=("failure_any", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )

    total = pd.DataFrame(
        {
            "classe": ["ALL"],
            "n": [len(df)],
            "latent_dist_mean": [df["latent_dist"].mean()],
            "latent_dist_std": [df["latent_dist"].std()],
            "PSNR_mean": [df["PSNR"].mean()],
            "PSNR_std": [df["PSNR"].std()],
            "SSIM_mean": [df["SSIM"].mean()],
            "SSIM_std": [df["SSIM"].std()],
            "MAE_mean": [df["MAE"].mean()],
            "MAE_std": [df["MAE"].std()],
            "failure_rate": [100 * df["failure_any"].mean()],
        }
    )

    return pd.concat([summary, total], ignore_index=True)


def correlation_table(df):
    rows = []

    for metric in ["PSNR", "SSIM", "MAE"]:
        pearson = df["latent_dist"].corr(df[metric], method="pearson")
        spearman = df["latent_dist"].corr(df[metric], method="spearman")

        rows.append(
            {
                "metric": metric,
                "pearson_corr_with_latent_dist": pearson,
                "spearman_corr_with_latent_dist": spearman,
            }
        )

    return pd.DataFrame(rows)


def assign_decisions(df, accept_q=0.50, reject_q=0.85, score_col="latent_dist"):
    if not (0 <= accept_q < reject_q <= 1):
        raise ValueError("Require 0 <= accept_q < reject_q <= 1.")

    tau_accept = df[score_col].quantile(accept_q)
    tau_reject = df[score_col].quantile(reject_q)

    df = df.copy()

    df["decision"] = np.select(
        [
            df[score_col] <= tau_accept,
            df[score_col] > tau_reject,
        ],
        [
            "accept",
            "reject",
        ],
        default="verify",
    )

    return df, tau_accept, tau_reject


def decision_summary(df):
    order = ["accept", "verify", "reject"]

    summary = (
        df.groupby("decision")
        .agg(
            n=("decision", "size"),
            percentage=("decision", lambda x: 100 * len(x) / len(df)),
            latent_dist_mean=("latent_dist", "mean"),
            latent_dist_std=("latent_dist", "std"),
            PSNR_mean=("PSNR", "mean"),
            PSNR_std=("PSNR", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            failure_rate=("failure_any", lambda x: 100 * x.mean()),
            failure_psnr_rate=("failure_psnr", lambda x: 100 * x.mean()),
            failure_ssim_rate=("failure_ssim", lambda x: 100 * x.mean()),
            failure_mae_rate=("failure_mae", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )

    summary["decision"] = pd.Categorical(
        summary["decision"], categories=order, ordered=True
    )

    return summary.sort_values("decision").reset_index(drop=True)


def policy_analysis(df):
    policies = {
        "conservative": (0.25, 0.75),
        "moderate": (0.50, 0.85),
        "permissive": (0.75, 0.95),
    }

    rows = []
    summaries = {}

    for policy_name, (accept_q, reject_q) in policies.items():
        df_dec, tau_accept, tau_reject = assign_decisions(
            df,
            accept_q=accept_q,
            reject_q=reject_q,
        )

        summary = decision_summary(df_dec)
        summaries[policy_name] = summary

        row = {
            "policy": policy_name,
            "accept_quantile": accept_q,
            "reject_quantile": reject_q,
            "tau_accept": tau_accept,
            "tau_reject": tau_reject,
        }

        for decision in ["accept", "verify", "reject"]:
            sub = summary[summary["decision"] == decision]

            if len(sub) == 0:
                row[f"{decision}_percentage"] = np.nan
                row[f"{decision}_MAE_mean"] = np.nan
                row[f"{decision}_PSNR_mean"] = np.nan
                row[f"{decision}_SSIM_mean"] = np.nan
                row[f"{decision}_failure_rate"] = np.nan
            else:
                row[f"{decision}_percentage"] = float(sub["percentage"].iloc[0])
                row[f"{decision}_MAE_mean"] = float(sub["MAE_mean"].iloc[0])
                row[f"{decision}_PSNR_mean"] = float(sub["PSNR_mean"].iloc[0])
                row[f"{decision}_SSIM_mean"] = float(sub["SSIM_mean"].iloc[0])
                row[f"{decision}_failure_rate"] = float(sub["failure_rate"].iloc[0])

        rows.append(row)

    return pd.DataFrame(rows), summaries


def risk_coverage_curve(df, score_col="latent_dist", risk_col="MAE"):
    dff = df.sort_values(score_col, ascending=True).reset_index(drop=True)

    rows = []
    n = len(dff)

    for k in range(1, n + 1):
        accepted = dff.iloc[:k]

        rows.append(
            {
                "coverage": k / n,
                "n_accepted": k,
                "threshold": accepted[score_col].max(),
                "risk_MAE": accepted[risk_col].mean(),
                "PSNR_mean": accepted["PSNR"].mean(),
                "SSIM_mean": accepted["SSIM"].mean(),
                "failure_rate": 100 * accepted["failure_any"].mean(),
            }
        )

    return pd.DataFrame(rows)


def plot_risk_coverage(curve, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(curve["coverage"], curve["risk_MAE"])
    plt.xlabel("Coverage: fraction of accepted samples")
    plt.ylabel("Mean MAE among accepted samples")
    plt.title("Risk-coverage curve")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_coverage_MAE.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(curve["coverage"], curve["failure_rate"])
    plt.xlabel("Coverage: fraction of accepted samples")
    plt.ylabel("Failure rate among accepted samples (%)")
    plt.title("Failure-coverage curve")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_coverage.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(curve["coverage"], curve["PSNR_mean"])
    plt.xlabel("Coverage: fraction of accepted samples")
    plt.ylabel("Mean PSNR among accepted samples")
    plt.title("PSNR-coverage curve")
    plt.tight_layout()
    plt.savefig(out_dir / "psnr_coverage.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(curve["coverage"], curve["SSIM_mean"])
    plt.xlabel("Coverage: fraction of accepted samples")
    plt.ylabel("Mean SSIM among accepted samples")
    plt.title("SSIM-coverage curve")
    plt.tight_layout()
    plt.savefig(out_dir / "ssim_coverage.png", dpi=300)
    plt.close()


def plot_decision_summary(summary, out_dir, policy_name):
    labels = summary["decision"].astype(str)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, summary["percentage"])
    plt.ylabel("Samples (%)")
    plt.xlabel("Qualification decision")
    plt.title(f"Decision distribution - {policy_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"decision_distribution_{policy_name}.png", dpi=300)
    plt.close()

    for metric in ["PSNR_mean", "SSIM_mean", "MAE_mean", "failure_rate"]:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, summary[metric])
        plt.ylabel(metric.replace("_", " "))
        plt.xlabel("Qualification decision")
        plt.title(f"{metric.replace('_', ' ')} - {policy_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_{policy_name}.png", dpi=300)
        plt.close()


def plot_scenario_summary(summary, out_dir):
    scenario_df = summary[summary["classe"] != "ALL"].copy()

    for metric in ["PSNR", "SSIM", "MAE", "latent_dist", "failure_rate"]:
        mean_col = f"{metric}_mean"

        if metric == "failure_rate":
            values = scenario_df["failure_rate"]
            ylabel = "Failure rate (%)"
        else:
            values = scenario_df[mean_col]
            ylabel = metric

        plt.figure(figsize=(7, 4))
        plt.bar(scenario_df["classe"], values)
        plt.ylabel(ylabel)
        plt.xlabel("Longitudinal scenario")
        plt.title(f"{ylabel} by longitudinal scenario")
        plt.tight_layout()
        plt.savefig(out_dir / f"scenario_{metric}.png", dpi=300)
        plt.close()


def plot_latent_distance_scatter(df, out_dir):
    for metric in ["PSNR", "SSIM", "MAE"]:
        plt.figure(figsize=(6, 4))

        for cls in sorted(df["classe"].unique()):
            sub = df[df["classe"] == cls]
            plt.scatter(
                sub["latent_dist"],
                sub[metric],
                alpha=0.65,
                s=18,
                label=cls,
            )

        plt.xlabel("Latent distance")
        plt.ylabel(metric)
        plt.title(f"Latent distance vs {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"latent_dist_vs_{metric}.png", dpi=300)
        plt.close()


def run_virtual_treatment_analysis(
    csv_path,
    out_dir="analysis_results",
    policy="moderate",
    save_outputs=True,
    make_plots=True,
):
    """
    Main function.

    Parameters
    ----------
    csv_path : str or Path
        Path to metrics.csv.

    out_dir : str or Path
        Directory where tables and figures will be saved.

    policy : str
        One of: "conservative", "moderate", "permissive".

    save_outputs : bool
        If True, save CSV tables and per-sample decisions.

    make_plots : bool
        If True, save PNG figures.

    Returns
    -------
    results : dict
        Dictionary containing all dataframes and thresholds.
    """

    policy_quantiles = {
        "conservative": (0.25, 0.75),
        "moderate": (0.50, 0.85),
        "permissive": (0.75, 0.95),
    }

    if policy not in policy_quantiles:
        raise ValueError(
            f"Unknown policy '{policy}'. "
            f"Choose one of {list(policy_quantiles.keys())}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(csv_path)
    df = add_failure_labels(df)

    scen_summary = scenario_summary(df)
    corr = correlation_table(df)
    policies, policy_summaries = policy_analysis(df)
    curve = risk_coverage_curve(df)

    accept_q, reject_q = policy_quantiles[policy]
    df_dec, tau_accept, tau_reject = assign_decisions(
        df,
        accept_q=accept_q,
        reject_q=reject_q,
    )

    main_decision_summary = decision_summary(df_dec)

    if save_outputs:
        scen_summary.to_csv(out_dir / "scenario_summary.csv", index=False)
        corr.to_csv(out_dir / "latent_distance_correlations.csv", index=False)
        policies.to_csv(out_dir / "policy_summary.csv", index=False)
        curve.to_csv(out_dir / "risk_coverage_curve.csv", index=False)

        for policy_name, summary in policy_summaries.items():
            summary.to_csv(
                out_dir / f"decision_summary_{policy_name}.csv",
                index=False,
            )

        df_dec.to_csv(
            out_dir / f"samples_with_decisions_{policy}.csv",
            index=False,
        )

    if make_plots:
        plot_scenario_summary(scen_summary, out_dir)
        plot_latent_distance_scatter(df, out_dir)
        plot_risk_coverage(curve, out_dir)

        for policy_name, summary in policy_summaries.items():
            plot_decision_summary(summary, out_dir, policy_name)

    results = {
        "df": df,
        "df_with_decisions": df_dec,
        "scenario_summary": scen_summary,
        "correlations": corr,
        "policy_summary": policies,
        "decision_summaries": policy_summaries,
        "main_decision_summary": main_decision_summary,
        "risk_coverage_curve": curve,
        "tau_accept": tau_accept,
        "tau_reject": tau_reject,
        "policy": policy,
        "out_dir": out_dir,
    }

    print(f"Loaded {len(df)} samples")
    print(f"Policy: {policy}")
    print(f"tau_accept = {tau_accept:.4f}")
    print(f"tau_reject = {tau_reject:.4f}")
    print(f"Outputs saved in: {out_dir.resolve()}")

    return results

if __name__ == "__main__":
    results = run_virtual_treatment_analysis(
        csv_path="/Users/francescodifeola/Desktop/metrics.csv",
        out_dir="/Users/francescodifeola/Desktop/analysis_results",
        policy="moderate",
    )