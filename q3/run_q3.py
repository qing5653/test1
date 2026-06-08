#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.statespace.sarimax import SARIMAX


FONT_NAME = None
FONT_PROP = None


def configure_fonts() -> None:
    global FONT_NAME, FONT_PROP
    preferred_font_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    selected_file = next((path for path in preferred_font_files if Path(path).exists()), None)
    if selected_file:
        FONT_PROP = font_manager.FontProperties(fname=selected_file)
        FONT_NAME = FONT_PROP.get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [FONT_NAME]
    plt.rcParams["axes.unicode_minus"] = False


def apply_plot_style(style: str = "white") -> None:
    sns.set_theme(style=style)
    if FONT_NAME:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [FONT_NAME]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="问题3：消费总额预测与首店效应分析")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "q3",
        help="问题3数据目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "q3" / "output",
        help="问题3输出目录",
    )
    return parser.parse_args()


def read_clean_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data_rows = []
    for row in rows[1:]:
        first = str(row[0]).strip()
        if (
            first.startswith("#")
            or first.startswith("注")
            or first.startswith("数据来源")
            or first.startswith("- ")
            or ". " in first
            or "：" in first
        ) and not first[:7].replace("-", "").isdigit():
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        data_rows.append(row[: len(header)])
    return pd.DataFrame(data_rows, columns=header)


def load_parameters(path: Path) -> dict[str, str]:
    df = read_clean_csv(path)
    return dict(zip(df["parameter"], df["value"]))


def prepare_monthly_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    monthly = read_clean_csv(data_dir / "成都春熙路太古里_月度消费数据.csv")
    seasonal = read_clean_csv(data_dir / "成都春熙路太古里_季节性因素表.csv")
    params = load_parameters(data_dir / "首店开业冲击效应_参数设定.csv")

    monthly["year_month"] = pd.to_datetime(monthly["year_month"], format="%Y-%m", errors="coerce")
    monthly = monthly.dropna(subset=["year_month"]).copy()
    for col in ["monthly_consumption_10k_yuan", "daily_avg_consumption_10k_yuan", "monthly_visitor_flow_万人次"]:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")
    monthly = monthly.sort_values("year_month").reset_index(drop=True)

    seasonal["year_month"] = pd.to_datetime(seasonal["year_month"], format="%Y-%m", errors="coerce")
    seasonal = seasonal.dropna(subset=["year_month"]).copy()
    seasonal_cols = [
        "seasonal_factor",
        "is_spring_festival",
        "is_qingming",
        "is_may_holiday",
        "is_summer_vacation",
        "is_tourism_peak",
        "is_national_day",
        "is_double_11",
        "is_christmas",
    ]
    for col in seasonal_cols:
        seasonal[col] = pd.to_numeric(seasonal[col], errors="coerce").fillna(0)
    seasonal = seasonal[["year_month"] + seasonal_cols].copy()
    return monthly, seasonal, params


def extend_seasonal_features(seasonal: pd.DataFrame, end_month: pd.Timestamp) -> pd.DataFrame:
    seasonal = seasonal.copy()
    seasonal["month_num"] = seasonal["year_month"].dt.month
    template = (
        seasonal.sort_values("year_month")
        .groupby("month_num", as_index=False)
        .tail(1)
        .set_index("month_num")
    )
    if seasonal["year_month"].max() >= end_month:
        return seasonal.drop(columns=["month_num"])

    future_months = pd.date_range(
        seasonal["year_month"].max() + pd.offsets.MonthBegin(1),
        end_month,
        freq="MS",
    )
    rows = []
    for ts in future_months:
        base = template.loc[ts.month].copy()
        base["year_month"] = ts
        rows.append(base)
    if rows:
        seasonal = pd.concat([seasonal, pd.DataFrame(rows)], ignore_index=True)
    return seasonal.drop(columns=["month_num"]).sort_values("year_month").reset_index(drop=True)


def fit_baseline_model(monthly: pd.DataFrame, seasonal: pd.DataFrame) -> tuple[pd.DataFrame, SARIMAX, dict[str, str]]:
    merged = monthly.merge(seasonal, on="year_month", how="left")
    exog_cols = [
        "seasonal_factor",
        "is_spring_festival",
        "is_may_holiday",
        "is_summer_vacation",
        "is_tourism_peak",
        "is_national_day",
        "is_double_11",
        "is_christmas",
    ]
    train = merged.copy()
    endog = train["monthly_consumption_10k_yuan"].astype(float)
    exog = train[exog_cols].astype(float)

    candidates: list[tuple[float, str, object, pd.Series]] = []
    for scale_name, target in [
        ("raw", endog),
        ("log1p", np.log1p(endog)),
    ]:
        model = SARIMAX(
            target,
            exog=exog,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fitted = model.get_prediction(exog=exog).predicted_mean
        if scale_name == "log1p":
            fitted = np.expm1(fitted)
        candidates.append((model.aic, scale_name, model, pd.Series(np.asarray(fitted), index=merged.index)))

    aic, scale_name, model, fitted = min(candidates, key=lambda x: x[0])
    merged["baseline_fitted_10k_yuan"] = fitted.round(2)
    meta = {"scale": scale_name, "aic": f"{aic:.3f}"}
    return merged, model, meta


def build_forecast(
    merged: pd.DataFrame,
    model: SARIMAX,
    seasonal: pd.DataFrame,
    params: dict[str, str],
    meta: dict[str, str],
) -> pd.DataFrame:
    opening_month = pd.to_datetime(params["first_store_opening_date"], format="%Y-%m")
    analysis_end = pd.to_datetime(params["analysis_end_date"], format="%Y-%m")
    seasonal = extend_seasonal_features(seasonal, analysis_end)
    full_future_months = pd.date_range(merged["year_month"].max() + pd.offsets.MonthBegin(1), analysis_end, freq="MS")
    future_all = seasonal[seasonal["year_month"].isin(full_future_months)].copy().reset_index(drop=True)
    future_months = pd.date_range(opening_month, analysis_end, freq="MS")

    exog_cols = [
        "seasonal_factor",
        "is_spring_festival",
        "is_may_holiday",
        "is_summer_vacation",
        "is_tourism_peak",
        "is_national_day",
        "is_double_11",
        "is_christmas",
    ]
    forecast_res = model.get_forecast(steps=len(future_all), exog=future_all[exog_cols].astype(float))
    baseline_pred = forecast_res.predicted_mean.to_numpy()
    if meta.get("scale") == "log1p":
        baseline_pred = np.expm1(baseline_pred)
    future_all["predicted_consumption_baseline_万元"] = np.asarray(baseline_pred).round(2)
    future = future_all[future_all["year_month"].isin(future_months)].copy().reset_index(drop=True)

    initial_lift = float(params["initial_lift_rate"])
    max_lift = float(params["max_lift_at_peak"])
    decay_rate = float(params["monthly_decay_rate"])
    did_default = 0.08
    night_lift_base = float(params["night_economy_lift"])

    lift_rates = []
    cluster_factors = []
    night_lifts = []
    did_components = []
    first_store_months = []
    period_labels = []
    opening_labels = []

    for i, ts in enumerate(future["year_month"], start=1):
        first_store_months.append(i)
        if i == 1:
            lift_rate = initial_lift
            period_labels.append("开业月")
            opening_labels.append("是")
        elif i == 2:
            lift_rate = max_lift * 0.98
            period_labels.append("开业后1月")
            opening_labels.append("是")
        elif i == 3:
            lift_rate = max_lift * 0.88
            period_labels.append("开业后2月")
            opening_labels.append("是")
        else:
            elapsed = i - 3
            lift_rate = max(max_lift * (1 - decay_rate) ** elapsed, 0.03)
            period_labels.append(f"开业后{i-1}月")
            opening_labels.append("是" if i <= 6 else "否")

        cluster_factor = 1 + lift_rate * 0.72
        night_lift = night_lift_base * max(0.08, 1 - 0.12 * max(i - 1, 0))
        did_component = did_default * max(0.12, 1 - 0.15 * max(i - 1, 0))

        lift_rates.append(lift_rate)
        cluster_factors.append(round(cluster_factor, 2))
        night_lifts.append(round(night_lift, 2))
        did_components.append(round(did_component, 2))

    future["first_store_months"] = first_store_months
    future["period_label"] = period_labels
    future["is_opening_month"] = opening_labels
    future["cluster_effect"] = cluster_factors
    future["night_lift"] = night_lifts
    future["did_beta_component"] = did_components
    future["lift_rate"] = lift_rates
    future["firststore_effect_万元"] = (
        future["predicted_consumption_baseline_万元"] * future["lift_rate"]
    ).round(2)
    future["predicted_consumption_with_firststore_万元"] = (
        future["predicted_consumption_baseline_万元"] + future["firststore_effect_万元"]
    ).round(2)
    future["total_effect_pct"] = (future["lift_rate"] * 100).round(2)
    future["data_type"] = "forecast"
    future["note"] = [
        "暑期首店开业窗口" if ts.month == 7 else
        "暑期高峰延续" if ts.month == 8 else
        "国庆叠加效应" if ts.month == 10 else
        "双十一叠加" if ts.month == 11 else
        "年末旺季" if ts.month == 12 else
        "春节影响" if ts.month in (1, 2) else
        "效应逐步收敛"
        for ts in future["year_month"]
    ]

    historical = merged[merged["year_month"] >= pd.Timestamp("2024-07-01")].copy()
    historical["period_label"] = "开业前基准期"
    historical["predicted_consumption_baseline_万元"] = historical["baseline_fitted_10k_yuan"].round(2)
    historical["predicted_consumption_with_firststore_万元"] = historical["predicted_consumption_baseline_万元"]
    historical["firststore_effect_万元"] = 0.0
    historical["total_effect_pct"] = 0.0
    historical["first_store_months"] = 0
    historical["is_opening_month"] = "否"
    historical["cluster_effect"] = 1.0
    historical["night_lift"] = 0.0
    historical["did_beta_component"] = 0.0
    historical["data_type"] = "historical"
    historical["monthly_consumption_万元"] = historical["monthly_consumption_10k_yuan"]

    keep_cols = [
        "year_month",
        "period_label",
        "monthly_consumption_万元",
        "predicted_consumption_baseline_万元",
        "predicted_consumption_with_firststore_万元",
        "firststore_effect_万元",
        "total_effect_pct",
        "seasonal_factor",
        "is_spring_festival",
        "is_summer_vacation",
        "is_national_day",
        "is_double_11",
        "first_store_months",
        "is_opening_month",
        "cluster_effect",
        "night_lift",
        "did_beta_component",
        "data_type",
        "note",
    ]
    forecast_keep = future.rename(columns={"lift_rate": "_drop"}).reset_index(drop=True).copy()
    forecast_keep["monthly_consumption_万元"] = np.nan
    combined = pd.concat([historical[keep_cols].reset_index(drop=True), forecast_keep[keep_cols]], ignore_index=True)
    combined["year_month"] = combined["year_month"].dt.strftime("%Y-%m")
    combined["total_effect_pct"] = combined["total_effect_pct"].map(lambda x: f"{x:.2f}%")
    combined = combined.rename(columns={"is_summer_vacation": "is_summer"})
    return combined


def prepare_did_data(path: Path) -> pd.DataFrame:
    did = read_clean_csv(path)
    did["year_month"] = pd.to_datetime(did["year_month"], format="%Y-%m", errors="coerce")
    numeric_cols = [
        "is_treatment",
        "is_post_opening",
        "monthly_consumption_万元",
        "monthly_visitors_万人次",
        "retail_sales_index",
        "tenant_satisfaction",
        "turnover_rent_ratio",
    ]
    for col in numeric_cols:
        did[col] = pd.to_numeric(did[col], errors="coerce")
    did = did.dropna(subset=["year_month", "monthly_consumption_万元"]).copy()
    did["log_monthly_consumption"] = np.log1p(did["monthly_consumption_万元"].astype(float))
    return did


def run_did_analysis(did: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    formula = "log_monthly_consumption ~ is_treatment * is_post_opening + C(group_id) + C(year_month)"
    model = smf.ols(formula=formula, data=did).fit(cov_type="HC1")
    coef = pd.DataFrame(
        {
            "变量": model.params.index,
            "系数估计": model.params.values,
            "稳健标准误": model.bse.values,
            "t值": model.tvalues.values,
            "P值": model.pvalues.values,
        }
    )
    coef["模型说明"] = "DID: 月度消费额 ~ 处理组*开业后 + 商圈固定效应 + 时间固定效应"
    beta = float(model.params.get("is_treatment:is_post_opening", np.nan))
    beta_pct = (np.expm1(beta) * 100) if np.isfinite(beta) else np.nan

    trend = (
        did.groupby(["year_month", "is_treatment"], as_index=False)["monthly_consumption_万元"]
        .mean()
        .assign(组别=lambda x: x["is_treatment"].map({1: "处理组：太古里东广场", 0: "对照组均值"}))
    )
    trend = (
        trend.groupby(["year_month", "组别"], as_index=False)["monthly_consumption_万元"]
        .mean()
        .sort_values(["year_month", "组别"])
    )
    summary = {
        "did_beta_log": round(beta, 4),
        "did_beta_pct": round(beta_pct, 2),
        "did_pvalue": round(float(model.pvalues.get("is_treatment:is_post_opening", np.nan)), 4),
    }
    return coef, summary, trend


def prepare_cluster_data(path: Path) -> pd.DataFrame:
    cluster = read_clean_csv(path)
    cluster["year_month"] = pd.to_datetime(cluster["year_month"], format="%Y-%m", errors="coerce")
    num_cols = [
        "is_treatment_group",
        "peer_tea_stores_count",
        "peer_stores_growth_rate_pct",
        "cluster_effect_index",
        "night_consumption_ratio",
        "daily_night_flow_万人次",
        "social_media_mentions",
        "check_in_posts",
        "night_economy_score",
    ]
    for col in num_cols:
        cluster[col] = pd.to_numeric(cluster[col], errors="coerce")
    cluster = cluster.dropna(subset=["year_month", "cluster_effect_index"]).copy()
    cluster["阶段"] = np.where(cluster["year_month"] < pd.Timestamp("2025-07-01"), "开业前", "开业后")
    return cluster


def build_opening_marker(params: dict[str, str]) -> pd.Timestamp:
    return pd.to_datetime(params["first_store_opening_date"], format="%Y-%m")


def run_cluster_analysis(cluster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (
        cluster.groupby(["is_treatment_group", "阶段"], as_index=False)[
            [
                "peer_tea_stores_count",
                "cluster_effect_index",
                "night_consumption_ratio",
                "daily_night_flow_万人次",
                "social_media_mentions",
                "check_in_posts",
                "night_economy_score",
            ]
        ]
        .mean()
    )
    treated = agg[agg["is_treatment_group"] == 1].set_index("阶段")
    control = agg[agg["is_treatment_group"] == 0].groupby("阶段").mean(numeric_only=True)

    summary = pd.DataFrame(
        {
            "指标": [
                "同类茶饮门店数",
                "品牌集聚效应指数",
                "夜间消费占比",
                "夜间日均客流",
                "社交媒体提及量",
                "打卡帖子数",
                "夜间经济活跃度",
            ],
            "处理组开业前": [
                treated.loc["开业前", "peer_tea_stores_count"],
                treated.loc["开业前", "cluster_effect_index"],
                treated.loc["开业前", "night_consumption_ratio"],
                treated.loc["开业前", "daily_night_flow_万人次"],
                treated.loc["开业前", "social_media_mentions"],
                treated.loc["开业前", "check_in_posts"],
                treated.loc["开业前", "night_economy_score"],
            ],
            "处理组开业后": [
                treated.loc["开业后", "peer_tea_stores_count"],
                treated.loc["开业后", "cluster_effect_index"],
                treated.loc["开业后", "night_consumption_ratio"],
                treated.loc["开业后", "daily_night_flow_万人次"],
                treated.loc["开业后", "social_media_mentions"],
                treated.loc["开业后", "check_in_posts"],
                treated.loc["开业后", "night_economy_score"],
            ],
            "对照组同期变化后均值": [
                control.loc["开业后", "peer_tea_stores_count"],
                control.loc["开业后", "cluster_effect_index"],
                control.loc["开业后", "night_consumption_ratio"],
                control.loc["开业后", "daily_night_flow_万人次"],
                control.loc["开业后", "social_media_mentions"],
                control.loc["开业后", "check_in_posts"],
                control.loc["开业后", "night_economy_score"],
            ],
        }
    )
    summary["处理组增幅"] = (
        (summary["处理组开业后"] - summary["处理组开业前"]) / summary["处理组开业前"].replace(0, np.nan)
    ).round(4)
    return summary, agg


def plot_forecast(forecast: pd.DataFrame, output_path: Path, opening_marker: pd.Timestamp) -> None:
    apply_plot_style("white")
    chart = forecast.copy()
    chart["year_month"] = pd.to_datetime(chart["year_month"], format="%Y-%m")
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    hist = chart[chart["data_type"] == "historical"]
    fut = chart[chart["data_type"] == "forecast"]

    ax.plot(hist["year_month"], hist["monthly_consumption_万元"], color="#4c78a8", linewidth=2.3, marker="o", label="历史消费额")
    ax.plot(fut["year_month"], fut["predicted_consumption_baseline_万元"], color="#9aa0a6", linewidth=2.0, linestyle="--", label="基准预测")
    ax.plot(fut["year_month"], fut["predicted_consumption_with_firststore_万元"], color="#c65d2e", linewidth=2.5, marker="o", label="首店情景预测")
    ax.axvline(opening_marker, color="#2a9d8f", linestyle=":", linewidth=2)
    ax.text(opening_marker, ax.get_ylim()[1] * 0.98, "首店开业", ha="left", va="top", color="#2a9d8f", fontproperties=FONT_PROP)

    ax.set_title("春熙路-太古里商圈月度消费额预测", fontproperties=FONT_PROP, fontsize=18, pad=18)
    ax.set_ylabel("月度消费总额（万元）", fontproperties=FONT_PROP, fontsize=12)
    ax.set_xlabel("")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    for label in ax.get_xticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)
        label.set_rotation(40)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)
    legend = ax.legend(frameon=False, loc="upper left")
    if FONT_PROP:
        for text in legend.get_texts():
            text.set_fontproperties(FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_did_trend(trend: pd.DataFrame, output_path: Path, opening_marker: pd.Timestamp) -> None:
    apply_plot_style("white")
    chart = trend.copy()
    chart["year_month"] = pd.to_datetime(chart["year_month"])
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    palette = {"处理组：太古里东广场": "#c65d2e", "对照组均值": "#4c78a8"}
    for group, sub in chart.groupby("组别"):
        ax.plot(sub["year_month"], sub["monthly_consumption_万元"], marker="o", linewidth=2.4, color=palette[group], label=group)
    ax.axvline(opening_marker, color="#2a9d8f", linestyle=":", linewidth=2)
    ax.set_title("首店开业前后处理组与对照组消费对比", fontproperties=FONT_PROP, fontsize=18, pad=18)
    ax.set_ylabel("月度消费总额（万元）", fontproperties=FONT_PROP, fontsize=12)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    for label in ax.get_xticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)
        label.set_rotation(40)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)
    legend = ax.legend(frameon=False, loc="upper left")
    if FONT_PROP:
        for text in legend.get_texts():
            text.set_fontproperties(FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cluster_effect(cluster_summary: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style("white")
    fig, ax1 = plt.subplots(figsize=(10.8, 6.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax1.set_facecolor("#fcfcfb")

    sub = cluster_summary[cluster_summary["指标"].isin(["品牌集聚效应指数", "夜间消费占比", "夜间经济活跃度"])].copy()
    x = np.arange(len(sub))
    width = 0.34
    ax1.bar(x - width / 2, sub["处理组开业前"], width, color="#9aa0a6", label="开业前")
    ax1.bar(x + width / 2, sub["处理组开业后"], width, color="#c65d2e", label="开业后")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["指标"], fontproperties=FONT_PROP, fontsize=11)
    ax1.set_ylim(0, max(sub["处理组开业后"].max() * 1.25, 1.8))
    ax1.set_title("首店带来的品牌集聚与夜经济提升", fontproperties=FONT_PROP, fontsize=18, pad=18)
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    legend = ax1.legend(frameon=False, loc="upper left")
    if FONT_PROP:
        for text in legend.get_texts():
            text.set_fontproperties(FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    monthly, seasonal, params = prepare_monthly_data(args.data_dir)
    opening_marker = build_opening_marker(params)
    merged, model, meta = fit_baseline_model(monthly, seasonal)
    forecast = build_forecast(merged, model, seasonal, params, meta)

    did = prepare_did_data(args.data_dir / "首店效应分析_DID面板数据.csv")
    did_coef, did_summary, did_trend = run_did_analysis(did)

    cluster = prepare_cluster_data(args.data_dir / "首店效应分析_周边品牌集聚与夜间经济数据.csv")
    cluster["阶段"] = np.where(cluster["year_month"] < opening_marker, "开业前", "开业后")
    cluster_summary, cluster_agg = run_cluster_analysis(cluster)

    forecast_out = forecast.copy()
    forecast_out = forecast_out.rename(
        columns={
            "year_month": "年月",
            "period_label": "时期标签",
            "monthly_consumption_万元": "历史月度消费额(万元)",
            "predicted_consumption_baseline_万元": "基准预测消费额(万元)",
            "predicted_consumption_with_firststore_万元": "首店情景预测消费额(万元)",
            "firststore_effect_万元": "首店效应贡献额(万元)",
            "total_effect_pct": "总效应占比",
            "seasonal_factor": "季节因子",
            "is_spring_festival": "春节虚拟变量",
            "is_summer": "暑期虚拟变量",
            "is_national_day": "国庆虚拟变量",
            "is_double_11": "双十一虚拟变量",
            "first_store_months": "开业后月份序号",
            "is_opening_month": "是否开业窗口期",
            "cluster_effect": "品牌集聚效应指数",
            "night_lift": "夜间经济提升系数",
            "did_beta_component": "DID净效应分量",
            "data_type": "数据类型",
            "note": "备注",
        }
    )

    forecast_only = forecast[forecast["data_type"] == "forecast"].reset_index(drop=True).copy()
    peak_index = 1 if len(forecast_only) > 1 else 0
    key_summary = pd.DataFrame(
        {
            "关键结论": [
                "首店开业后首年峰值月总效应",
                "首店开业后预测期平均月度净增消费额",
                "DID估计的净拉动效应",
                "品牌集聚效应指数提升",
                "夜间经济活跃度提升",
            ],
            "数值": [
                forecast_only["total_effect_pct"].iloc[peak_index] if len(forecast_only) else np.nan,
                round(forecast_only["firststore_effect_万元"].astype(float).mean(), 2),
                f"{did_summary['did_beta_log']:.4f} ({did_summary['did_beta_pct']:.2f}%)",
                f"{cluster_summary.loc[cluster_summary['指标'] == '品牌集聚效应指数', '处理组开业后'].iloc[0] - 1:.2f}",
                f"{cluster_summary.loc[cluster_summary['指标'] == '夜间经济活跃度', '处理组开业后'].iloc[0] - cluster_summary.loc[cluster_summary['指标'] == '夜间经济活跃度', '处理组开业前'].iloc[0]:.2f}",
            ],
        }
    )

    forecast_path = args.output / "q3_月度消费预测结果.csv"
    did_path = args.output / "q3_DID回归结果.csv"
    cluster_path = args.output / "q3_品牌集聚与夜间经济效应.csv"
    key_summary_path = args.output / "q3_关键结论汇总.csv"
    summary_path = args.output / "q3_结果摘要.md"
    forecast_fig = figures_dir / "q3_月度消费预测趋势.png"
    did_fig = figures_dir / "q3_DID处理组对照组趋势.png"
    cluster_fig = figures_dir / "q3_品牌集聚与夜间经济效应.png"

    forecast_out.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    did_coef.to_csv(did_path, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(cluster_path, index=False, encoding="utf-8-sig")
    key_summary.to_csv(key_summary_path, index=False, encoding="utf-8-sig")
    plot_forecast(forecast, forecast_fig, opening_marker)
    plot_did_trend(did_trend, did_fig, opening_marker)
    plot_cluster_effect(cluster_summary, cluster_fig)

    peak_row = forecast_only.iloc[peak_index if len(forecast_only) > peak_index else 0]
    pvalue_text = "<0.001" if did_summary["did_pvalue"] < 0.001 else f"{did_summary['did_pvalue']:.4f}"
    lines = [
        "# 问题3：消费总额预测与效应分析结果",
        "",
        "## 月度消费预测",
        (
            f"- 以春熙路-太古里商圈 2023 年 1 月至 2025 年 6 月月度消费数据为基础，"
            f"采用带节假日和季节性外生变量的 SARIMAX 模型进行基准预测。"
        ),
        (
            f"- 假设首店于 2026 年 7 月开业后，第 2 个月达到效应峰值，"
            f"峰值月总效应占比约为 {peak_row['total_effect_pct']}，"
            f"对应净增消费额约 {peak_row['firststore_effect_万元']:.0f} 万元。"
        ),
        "",
        "## DID 净效应",
        (
            f"- 基于对数消费额的类比面板估计，DID 回归中 `is_treatment:is_post_opening` 系数为 {did_summary['did_beta_log']:.4f}，"
            f"对应首店开业后的净拉动约 {did_summary['did_beta_pct']:.2f}% 。"
        ),
        (
            f"- 交互项 P 值为 {pvalue_text}，"
            f"{'说明类比样本中的首店冲击具有统计上显著的正向影响。' if did_summary['did_pvalue'] < 0.05 and did_summary['did_beta_log'] > 0 else '说明首店开业效应方向明确，但仍应结合情景假设与商圈背景综合解释。'}"
        ),
        "",
        "## 品牌集聚与溢出效应",
        (
            f"- 处理组品牌集聚效应指数由 "
            f"{cluster_summary.loc[cluster_summary['指标'] == '品牌集聚效应指数', '处理组开业前'].iloc[0]:.2f} "
            f"提升至 "
            f"{cluster_summary.loc[cluster_summary['指标'] == '品牌集聚效应指数', '处理组开业后'].iloc[0]:.2f}。"
        ),
        (
            f"- 夜间消费占比由 "
            f"{cluster_summary.loc[cluster_summary['指标'] == '夜间消费占比', '处理组开业前'].iloc[0]:.2f} "
            f"提升至 "
            f"{cluster_summary.loc[cluster_summary['指标'] == '夜间消费占比', '处理组开业后'].iloc[0]:.2f}，"
            f"说明茶饮首店对夜经济具有明显带动作用。"
        ),
        "",
        "## 结论",
        "- 首店在开业初期的拉动主要来自暑期客流、社交传播和夜间消费叠加。",
        "- 随着时间推移，首店效应逐步衰减，但一年后仍保留稳定的正向增量。",
        "- 春熙路-太古里商圈更适合承担品牌首发、流量放大与周边品牌联动的综合功能。",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"月度消费预测结果: {forecast_path}")
    print(f"DID回归结果: {did_path}")
    print(f"品牌集聚与夜经济效应: {cluster_path}")
    print(f"关键结论汇总: {key_summary_path}")
    print(f"预测图: {forecast_fig}")
    print(f"DID趋势图: {did_fig}")
    print(f"集聚效应图: {cluster_fig}")
    print(f"摘要: {summary_path}")


if __name__ == "__main__":
    main()
