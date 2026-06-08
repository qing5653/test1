#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


POSITIVE_INDICATORS = {
    "customer_profile_score": "客群匹配度",
    "price_affordability_score": "消费能力匹配度",
    "daily_foot_traffic": "日均客流量",
    "transport_score": "交通便利度",
    "cost_effectiveness_score": "租金性价比",
    "night_economy_score": "夜间活跃度",
    "weekend_holiday_score": "周末节假日热度",
    "social_media_heat_index": "社交媒体热度",
}

NEGATIVE_INDICATORS = {
    "competition_intensity_index": "竞争强度",
    "rent_avg_yuan_sqm_month": "平均租金",
    "distance_to_metro_m": "距地铁距离",
}

SUBJECTIVE_WEIGHTS = {
    "customer_profile_score": 0.16,
    "price_affordability_score": 0.16,
    "daily_foot_traffic": 0.18,
    "transport_score": 0.14,
    "cost_effectiveness_score": 0.12,
    "night_economy_score": 0.07,
    "weekend_holiday_score": 0.07,
    "social_media_heat_index": 0.08,
    "competition_intensity_index": 0.07,
    "rent_avg_yuan_sqm_month": 0.03,
    "distance_to_metro_m": 0.02,
}

CHINESE_HEADERS = {
    "candidate_id": "候选区域编号",
    "candidate_name": "候选区域名称",
    "district": "所属行政区",
    "business_circle": "所属商圈",
    "area_type": "区域类型",
    "latitude": "纬度",
    "longitude": "经度",
    "data_source": "数据来源",
    "collection_method": "采集方式",
    "metro_station_name": "最近地铁站",
    "distance_to_metro_m": "距地铁站距离(米)",
    "metro_lines_count": "地铁线路数",
    "bus_stops_300m": "300米公交站点数",
    "is_near_main_road": "临主干道",
    "is_walking_street": "步行街属性",
    "pedestrian_accessibility_score": "步行可达性得分",
    "parking_lots_500m": "500米停车场数",
    "transport_score": "交通便利度得分",
    "has_university": "临近高校",
    "has_office_building": "临近写字楼",
    "has_shopping_mall": "临近购物中心",
    "has_tourist_attraction": "临近旅游景点",
    "young_population_ratio_pct": "年轻人口占比(%)",
    "daily_foot_traffic": "日均客流量",
    "night_economy_score": "夜间活跃度得分",
    "weekend_holiday_score": "周末节假日热度得分",
    "social_media_heat_index": "社交媒体热度指数",
    "customer_profile_score": "客群匹配度得分",
    "price_affordability_score": "消费能力匹配度得分",
    "rent_min_yuan_sqm_month": "最低租金(元/㎡/月)",
    "rent_max_yuan_sqm_month": "最高租金(元/㎡/月)",
    "rent_avg_yuan_sqm_month": "平均租金(元/㎡/月)",
    "available_area_min_sqm": "可选面积下限(㎡)",
    "available_area_max_sqm": "可选面积上限(㎡)",
    "rent_to_sales_ratio": "租售比",
    "lease_term_years": "常见租约年限",
    "cost_effectiveness_score": "租金性价比得分",
    "bubble_tea_stores_500m": "500米茶饮门店数",
    "coffee_sBrand_500m": "500米咖啡门店数",
    "dessert_brand_500m": "500米甜品门店数",
    "total_competitors_500m": "500米总竞争门店数",
    "bubble_tea_density": "茶饮密度等级",
    "competition_intensity_index": "竞争强度指数",
    "candidate_business_label": "候选区域标签",
    "TOPSIS贴近度": "TOPSIS贴近度",
    "综合排名": "综合排名",
    "推荐理由": "推荐理由",
    "指标名称": "指标名称",
    "指标含义": "指标含义",
    "主观权重": "主观权重",
    "熵权": "熵权",
    "组合权重": "组合权重",
    "推荐等级": "推荐等级",
    "综合评价展示分": "综合评价展示分",
}

FONT_NAME = None
FONT_PROP = None


def configure_fonts() -> None:
    global FONT_NAME, FONT_PROP
    preferred_font_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
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
    parser = argparse.ArgumentParser(description="问题2：成都茶饮首店选址优化")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "q2",
        help="问题2原始数据目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "q2" / "output",
        help="问题2输出目录",
    )
    return parser.parse_args()


def load_valid_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    ids = df["candidate_id"].astype(str)
    return df[ids.str.startswith("CD-C", na=False)].copy()


def to_bool_num(series: pd.Series) -> pd.Series:
    mapping = {"是": 1, "否": 0}
    return series.map(mapping).fillna(series)


def minmax_scale(series: pd.Series, positive: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    min_val = s.min()
    max_val = s.max()
    if np.isclose(max_val, min_val):
        return pd.Series(np.ones(len(s)), index=s.index)
    scaled = (s - min_val) / (max_val - min_val)
    return scaled if positive else 1 - scaled


def entropy_weights(matrix: pd.DataFrame) -> pd.Series:
    safe = matrix.copy().astype(float).fillna(0)
    col_sums = safe.sum(axis=0)
    proportion = safe.div(col_sums.where(col_sums != 0, np.nan), axis=1).fillna(0)
    n = len(proportion)
    k = 1.0 / np.log(n) if n > 1 else 0.0

    entropy = []
    for col in proportion.columns:
        p = proportion[col].to_numpy()
        valid = p > 0
        value = -k * np.sum(p[valid] * np.log(p[valid])) if valid.any() else 1.0
        entropy.append(value)
    entropy = pd.Series(entropy, index=proportion.columns)
    divergence = 1 - entropy
    if np.isclose(divergence.sum(), 0):
        return pd.Series(1 / len(divergence), index=divergence.index)
    return divergence / divergence.sum()


def combine_weights(entropy: pd.Series, subjective: dict[str, float], alpha: float = 0.6) -> pd.Series:
    subjective_series = pd.Series(subjective, dtype=float).reindex(entropy.index).fillna(0)
    subjective_series = subjective_series / subjective_series.sum()
    combined = alpha * subjective_series + (1 - alpha) * entropy
    return combined / combined.sum()


def build_wide_table(data_dir: Path) -> pd.DataFrame:
    base = load_valid_csv(data_dir / "成都候选区域清单.csv")
    traffic = load_valid_csv(data_dir / "成都交通便利度指标.csv")
    profile = load_valid_csv(data_dir / "成都客群匹配度指标.csv")
    rent = load_valid_csv(data_dir / "成都租金成本指标.csv")
    competition = load_valid_csv(data_dir / "成都竞争强度指标.csv")

    wide = base.merge(
        traffic.drop(columns=["candidate_name", "data_source", "collection_method"]),
        on="candidate_id",
        how="left",
    )
    wide = wide.merge(
        profile.drop(columns=["candidate_name", "data_source", "collection_method"]),
        on="candidate_id",
        how="left",
    )
    wide = wide.merge(
        rent.drop(columns=["candidate_name", "data_source", "collection_method"]),
        on="candidate_id",
        how="left",
    )
    wide = wide.merge(
        competition.drop(columns=["candidate_name", "data_source", "collection_method"]),
        on="candidate_id",
        how="left",
    )

    for col in ["is_near_main_road", "is_walking_street", "has_university", "has_office_building", "has_shopping_mall", "has_tourist_attraction"]:
        wide[col] = to_bool_num(wide[col])

    numeric_cols = [
        "distance_to_metro_m", "metro_lines_count", "bus_stops_300m", "pedestrian_accessibility_score",
        "parking_lots_500m", "transport_score", "young_population_ratio_pct", "daily_foot_traffic",
        "night_economy_score", "weekend_holiday_score", "social_media_heat_index", "customer_profile_score",
        "rent_min_yuan_sqm_month", "rent_max_yuan_sqm_month", "rent_avg_yuan_sqm_month",
        "available_area_min_sqm", "available_area_max_sqm", "rent_to_sales_ratio",
        "cost_effectiveness_score", "bubble_tea_stores_500m", "coffee_sBrand_500m",
        "dessert_brand_500m", "total_competitors_500m", "competition_intensity_index",
    ]
    for col in numeric_cols:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    # 30-40元客单价要求年轻客群之外还具备一定消费能力，避免纯校园低承受力区域被高估。
    price_affordability = (
        0.32 * wide["has_office_building"].astype(float)
        + 0.22 * wide["has_shopping_mall"].astype(float)
        + 0.14 * wide["has_tourist_attraction"].astype(float)
        + 0.18 * wide["customer_profile_score"].astype(float)
        + 0.14 * minmax_scale(wide["rent_avg_yuan_sqm_month"], positive=True)
    )
    campus_penalty = np.where(
        (wide["has_university"].astype(float) == 1)
        & (wide["has_office_building"].astype(float) == 0)
        & (wide["has_shopping_mall"].astype(float) == 0),
        0.18,
        0.0,
    )
    wide["price_affordability_score"] = (price_affordability - campus_penalty).clip(0, 1).round(4)
    wide["candidate_business_label"] = wide["business_circle"] + "｜" + wide["area_type"]
    return wide


def build_standard_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=wide.index)
    for col in POSITIVE_INDICATORS:
        matrix[col] = minmax_scale(wide[col], positive=True)
    for col in NEGATIVE_INDICATORS:
        matrix[col] = minmax_scale(wide[col], positive=False)
    return matrix


def topsis_score(matrix: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weighted = matrix.mul(weights, axis=1)
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return dist_worst / (dist_best + dist_worst)


def recommend_reason(row: pd.Series) -> str:
    parts = []
    if row["customer_profile_score"] >= 0.85:
        parts.append("目标年轻客群匹配度高")
    if row["price_affordability_score"] >= 0.75:
        parts.append("消费能力与30-40元客单价匹配")
    if row["daily_foot_traffic"] >= 70000:
        parts.append("日均客流量表现突出")
    if row["transport_score"] >= 0.88:
        parts.append("交通可达性优")
    if row["cost_effectiveness_score"] >= 0.85:
        parts.append("租金性价比较高")
    if row["competition_intensity_index"] <= 0.60:
        parts.append("竞争强度相对可控")
    if row["night_economy_score"] >= 0.85:
        parts.append("夜间消费活跃")
    return "；".join(parts[:4]) if parts else "综合表现均衡，适合作为候选点"


def build_dimension_score_table(wide: pd.DataFrame) -> pd.DataFrame:
    score_df = pd.DataFrame(
        {
            "candidate_id": wide["candidate_id"],
            "candidate_name": wide["candidate_name"],
            "business_circle": wide["business_circle"],
            "客流得分": minmax_scale(wide["daily_foot_traffic"], positive=True).round(2),
            "年轻人占比得分": minmax_scale(wide["young_population_ratio_pct"], positive=True).round(2),
            "消费能力匹配得分": pd.to_numeric(wide["price_affordability_score"], errors="coerce").round(2),
            "交通便利度得分": pd.to_numeric(wide["transport_score"], errors="coerce").round(2),
            "租金性价比得分": pd.to_numeric(wide["cost_effectiveness_score"], errors="coerce").round(2),
            "客群匹配度得分": pd.to_numeric(wide["customer_profile_score"], errors="coerce").round(2),
            "竞争强度得分(逆向)": (1 - pd.to_numeric(wide["competition_intensity_index"], errors="coerce")).round(2),
        }
    )
    score_df["综合评价展示分"] = (
        0.18 * score_df["客流得分"]
        + 0.10 * score_df["年轻人占比得分"]
        + 0.18 * score_df["消费能力匹配得分"]
        + 0.15 * score_df["交通便利度得分"]
        + 0.15 * score_df["租金性价比得分"]
        + 0.14 * score_df["客群匹配度得分"]
        + 0.15 * score_df["竞争强度得分(逆向)"]
    ).round(2)
    score_df = score_df.sort_values(
        ["综合评价展示分", "客群匹配度得分", "客流得分"], ascending=[False, False, False]
    ).reset_index(drop=True)
    return score_df


def build_business_circle_recommendations(result: pd.DataFrame) -> pd.DataFrame:
    representative = (
        result.sort_values(
            ["business_circle", "TOPSIS贴近度", "customer_profile_score", "transport_score"],
            ascending=[True, False, False, False],
        )
        .groupby("business_circle", as_index=False)
        .head(1)
        .copy()
        .reset_index(drop=True)
    )
    representative["商圈内排名"] = representative.groupby("business_circle")["TOPSIS贴近度"].rank(
        method="dense", ascending=False
    ).astype(int)
    representative["商圈推荐定位"] = [
        "品牌首发核心点" if circle == "春熙路-太古里"
        else "年轻客群性价比点" if circle == "建设路-万达"
        else "交通枢纽稳健点"
        for circle in representative["business_circle"]
    ]
    return representative


def plot_q2_ranking(result: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style("white")
    ordered = result.sort_values("TOPSIS贴近度", ascending=True).reset_index(drop=True)

    circle_palette = {
        "春熙路-太古里": "#c65d2e",
        "盐市口-天府广场": "#2a9d8f",
        "建设路-万达": "#4c78a8",
    }
    colors = [circle_palette.get(x, "#8f8f8f") for x in ordered["business_circle"]]

    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.barh(
        ordered["candidate_name"],
        ordered["TOPSIS贴近度"],
        color=colors,
        height=0.72,
        edgecolor="none",
    )

    ax.set_xlim(0, max(0.78, ordered["TOPSIS贴近度"].max() + 0.08))
    ax.set_xlabel("TOPSIS贴近度", fontproperties=FONT_PROP, fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("成都茶饮首店候选区域综合排序", fontproperties=FONT_PROP, fontsize=18, pad=18)
    ax.xaxis.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#b5b5b5")
    ax.tick_params(axis="y", length=0, pad=12, labelsize=12)
    ax.tick_params(axis="x", labelsize=11, colors="#555555")

    for label in ax.get_yticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)
    for label in ax.get_xticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)

    for i, (bar, value, circle) in enumerate(zip(bars, ordered["TOPSIS贴近度"], ordered["business_circle"])):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
            fontproperties=FONT_PROP,
        )
        ax.text(
            0.01,
            bar.get_y() + bar.get_height() / 2,
            f"#{len(ordered) - i}",
            va="center",
            ha="left",
            fontsize=10,
            color="#ffffff",
            fontproperties=FONT_PROP,
            bbox=dict(boxstyle="round,pad=0.18", fc=colors[i], ec="none"),
        )

    fig.text(
        0.125,
        0.94,
        "中高客单价茶饮品牌更偏好春熙路-太古里板块，建设路与盐市口形成次级候选梯队",
        fontsize=11,
        color="#666666",
        fontproperties=FONT_PROP,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_q2_business_circle_representatives(rep_df: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style("white")
    circle_col = "所属商圈" if "所属商圈" in rep_df.columns else "business_circle"
    score_col = "TOPSIS贴近度"
    name_col = "候选区域名称" if "候选区域名称" in rep_df.columns else "candidate_name"
    label_col = "商圈推荐定位"

    ordered = rep_df.sort_values(score_col, ascending=True).reset_index(drop=True)
    circle_palette = {
        "春熙路-太古里": "#c65d2e",
        "盐市口-天府广场": "#2a9d8f",
        "建设路-万达": "#4c78a8",
    }
    colors = [circle_palette.get(x, "#8f8f8f") for x in ordered[circle_col]]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.barh(
        ordered[circle_col],
        ordered[score_col],
        color=colors,
        height=0.55,
        edgecolor="none",
    )

    ax.set_xlim(0, max(0.78, ordered[score_col].max() + 0.10))
    ax.set_xlabel("代表点 TOPSIS贴近度", fontproperties=FONT_PROP, fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("三大商圈代表候选点对比", fontproperties=FONT_PROP, fontsize=18, pad=18)
    ax.xaxis.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#b5b5b5")
    ax.tick_params(axis="y", length=0, pad=10, labelsize=12)
    ax.tick_params(axis="x", labelsize=11, colors="#555555")

    for label in ax.get_yticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)
    for label in ax.get_xticklabels():
        if FONT_PROP:
            label.set_fontproperties(FONT_PROP)

    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(
            row[score_col] + 0.01,
            bar.get_y() + bar.get_height() / 2 + 0.12,
            f"{row[name_col]}  ({row[score_col]:.3f})",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
            fontproperties=FONT_PROP,
        )
        ax.text(
            row[score_col] + 0.01,
            bar.get_y() + bar.get_height() / 2 - 0.12,
            row[label_col],
            va="center",
            ha="left",
            fontsize=10,
            color="#777777",
            fontproperties=FONT_PROP,
        )

    fig.text(
        0.125,
        0.94,
        "在保留全局最优判断的同时，为不同商圈各选出一个代表点，便于论文形成分层推荐结论",
        fontsize=11,
        color="#666666",
        fontproperties=FONT_PROP,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_q2_top3_radar(display_scores: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style("white")
    top3 = display_scores.head(3).copy()
    metrics = ["客流得分", "消费能力匹配得分", "交通便利度得分", "租金性价比得分", "客群匹配度得分", "竞争强度得分(逆向)"]
    labels = metrics

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.4, 8.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("#fcfcfb")

    colors = ["#c65d2e", "#2a9d8f", "#4c78a8"]
    for idx, (_, row) in enumerate(top3.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2.2, label=row["候选区域名称"])
        ax.fill(angles, values, color=colors[idx], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=FONT_PROP, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontproperties=FONT_PROP, fontsize=9, color="#666666")
    ax.set_ylim(0, 1.0)
    ax.grid(color="#d9d9d9", linewidth=0.8)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Top3候选区域多维竞争力对比", fontproperties=FONT_PROP, fontsize=18, pad=24)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False, fontsize=10)
    if FONT_PROP:
        for text in legend.get_texts():
            text.set_fontproperties(FONT_PROP)

    fig.text(
        0.12,
        0.95,
        "雷达图用于比较 Top3 在客流、消费能力、交通、成本与竞争等维度上的差异",
        fontsize=11,
        color="#666666",
        fontproperties=FONT_PROP,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    wide = build_wide_table(args.data_dir)
    matrix = build_standard_matrix(wide)
    entropy = entropy_weights(matrix)
    weights = combine_weights(entropy, SUBJECTIVE_WEIGHTS, alpha=0.6)
    closeness = topsis_score(matrix, weights)
    display_scores = build_dimension_score_table(wide)

    result = wide.copy()
    result["TOPSIS贴近度"] = closeness.round(4)
    result["综合排名"] = result["TOPSIS贴近度"].rank(method="dense", ascending=False).astype(int)
    result = result.sort_values(["TOPSIS贴近度", "customer_profile_score", "transport_score"], ascending=[False, False, False]).reset_index(drop=True)
    result["推荐理由"] = result.apply(recommend_reason, axis=1)

    weights_df = pd.DataFrame(
        {
            "指标名称": list(weights.index),
            "指标含义": [POSITIVE_INDICATORS.get(col, NEGATIVE_INDICATORS.get(col)) for col in weights.index],
            "主观权重": [SUBJECTIVE_WEIGHTS[col] for col in weights.index],
            "熵权": [entropy[col] for col in weights.index],
            "组合权重": weights.values,
        }
    ).sort_values("组合权重", ascending=False)

    matrix_out = matrix.copy()
    matrix_out.insert(0, "candidate_id", wide["candidate_id"])
    matrix_out.insert(1, "candidate_name", wide["candidate_name"])
    matrix_out["TOPSIS贴近度"] = closeness.round(4)

    top3 = result.head(3).copy()
    top3["推荐等级"] = ["首选", "备选1", "备选2"]
    business_circle_reps = build_business_circle_recommendations(result)

    wide_path = args.output / "q2_成都候选区域宽表.csv"
    weights_path = args.output / "q2_指标权重.csv"
    matrix_path = args.output / "q2_标准化指标矩阵.csv"
    result_path = args.output / "q2_成都首店选址排序结果.csv"
    top3_path = args.output / "q2_Top3推荐区域.csv"
    rep_path = args.output / "q2_分商圈代表推荐.csv"
    summary_path = args.output / "q2_结果摘要.md"
    display_score_path = args.output / "q2_分项得分汇总表.csv"
    ranking_fig_path = figures_dir / "q2_候选区域综合排序.png"
    radar_fig_path = figures_dir / "q2_Top3多维对比雷达图.png"
    rep_fig_path = figures_dir / "q2_分商圈代表推荐对比.png"

    wide.rename(columns=CHINESE_HEADERS).to_csv(wide_path, index=False, encoding="utf-8-sig")
    weights_df.rename(columns=CHINESE_HEADERS).to_csv(weights_path, index=False, encoding="utf-8-sig")
    matrix_out.rename(columns=CHINESE_HEADERS).to_csv(matrix_path, index=False, encoding="utf-8-sig")
    result.rename(columns=CHINESE_HEADERS).to_csv(result_path, index=False, encoding="utf-8-sig")
    top3.rename(columns=CHINESE_HEADERS).to_csv(top3_path, index=False, encoding="utf-8-sig")
    business_circle_reps.rename(columns=CHINESE_HEADERS).to_csv(rep_path, index=False, encoding="utf-8-sig")
    display_scores.rename(columns=CHINESE_HEADERS).to_csv(display_score_path, index=False, encoding="utf-8-sig")
    plot_q2_ranking(result, ranking_fig_path)
    plot_q2_top3_radar(display_scores.rename(columns=CHINESE_HEADERS), radar_fig_path)
    plot_q2_business_circle_representatives(
        business_circle_reps.rename(columns=CHINESE_HEADERS), rep_fig_path
    )

    lines = [
        "# 问题2：成都茶饮首店选址优化结果",
        "",
        "## 模型方法",
        "基于候选区域级数据，构建客群匹配、客流潜力、交通可达、租金成本与竞争环境五大类指标，采用 AHP-熵权组合赋权与 TOPSIS 方法进行排序。",
        "",
        "## Top3 推荐区域",
    ]
    for _, row in top3.iterrows():
        lines.append(
            f"- {row['推荐等级']}：`{row['candidate_name']}`（{row['business_circle']}，贴近度 {row['TOPSIS贴近度']:.4f}）。{row['推荐理由']}。"
        )
    lines.extend(
        [
            "",
            "## 分商圈代表推荐",
        ]
    )
    for _, row in business_circle_reps.iterrows():
        lines.append(
            f"- `{row['business_circle']}`：推荐 `{row['candidate_name']}`（贴近度 {row['TOPSIS贴近度']:.4f}），定位为“{row['商圈推荐定位']}”。{row['推荐理由']}。"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "- 全局排序显示，中高客单价茶饮首店最优落位仍集中于春熙路-太古里板块。",
            "- 为避免结论过于集中，本文进一步给出分商圈代表点推荐，用于形成“最优点 + 备选商圈”的分层决策结构。",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"候选区域宽表: {wide_path}")
    print(f"排序结果: {result_path}")
    print(f"Top3推荐: {top3_path}")
    print(f"分商圈代表推荐: {rep_path}")
    print(f"指标权重: {weights_path}")
    print(f"分项得分汇总: {display_score_path}")
    print(f"排序图: {ranking_fig_path}")
    print(f"Top3雷达图: {radar_fig_path}")
    print(f"分商圈代表图: {rep_fig_path}")
    print(f"摘要: {summary_path}")


if __name__ == "__main__":
    main()
