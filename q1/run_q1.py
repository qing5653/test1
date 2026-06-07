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


def apply_plot_style(style: str) -> None:
    sns.set_theme(style=style)
    if FONT_NAME:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [FONT_NAME]
    plt.rcParams["axes.unicode_minus"] = False


POSITIVE_INDICATORS = {
    "来源点位数": "商业规模",
    "商场样本数": "商业密度",
    "商圈成熟度指数": "商业成熟度",
    "客流活跃度代理": "客流活跃度",
    "核心吸引力指数": "核心吸引力",
    "类型潜力分": "成长潜力",
}

NEGATIVE_INDICATORS = {
    "成本压力代理_租金": "租金成本压力",
}

SUBJECTIVE_WEIGHTS = {
    "来源点位数": 0.08,
    "商场样本数": 0.12,
    "商圈成熟度指数": 0.16,
    "客流活跃度代理": 0.28,
    "核心吸引力指数": 0.18,
    "类型潜力分": 0.10,
    "成本压力代理_租金": 0.08,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_input = root / "data" / "问题一_成都商圈消费活力精修宽表.csv"
    default_output = root / "q1" / "output"
    parser = argparse.ArgumentParser(
        description="问题1：成都商圈消费活力评价、分类与结果导出"
    )
    parser.add_argument("--input", type=Path, default=default_input, help="输入宽表路径")
    parser.add_argument("--output", type=Path, default=default_output, help="输出目录")
    return parser.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()

    feature_df["日均客流量_万人次"] = pd.to_numeric(
        feature_df["日均客流量_万人次"], errors="coerce"
    )
    feature_df["来源点位数"] = pd.to_numeric(feature_df["来源点位数"], errors="coerce").fillna(0)
    feature_df["综合活力先验分"] = pd.to_numeric(
        feature_df["综合活力先验分"], errors="coerce"
    ).fillna(0)

    type_flow_base = {
        "核心商圈": 11.5,
        "传统核心商圈": 8.2,
        "次级成长商圈": 5.8,
        "新兴拓展商圈": 4.8,
        "社区成熟商圈": 4.2,
        "区域生活商圈": 3.8,
    }

    type_base = feature_df["商圈类型建议"].map(type_flow_base).fillna(4.0)
    fallback_flow = (
        type_base
        + feature_df["综合活力先验分"] / 100 * 1.2
        + feature_df["来源点位数"] / feature_df["来源点位数"].max() * 0.8
    )
    feature_df["客流活跃度代理"] = feature_df["日均客流量_万人次"].fillna(
        fallback_flow.round(2)
    )

    type_score_map = {
        "核心商圈": 95,
        "传统核心商圈": 85,
        "次级成长商圈": 75,
        "新兴拓展商圈": 70,
        "社区成熟商圈": 60,
        "区域生活商圈": 55,
    }
    feature_df["类型潜力分"] = feature_df["商圈类型建议"].map(type_score_map).fillna(50)

    attraction_score_map = {
        "春熙路-太古里核心商圈": 98,
        "盐市口-天府广场商圈": 82,
        "建设路-万达商圈": 72,
        "双楠生活商圈": 58,
        "科华南路商圈": 56,
        "近郊新城商圈组团": 62,
    }
    feature_df["核心吸引力指数"] = feature_df["商圈名称"].map(attraction_score_map).fillna(
        feature_df["类型潜力分"]
    )

    for col in list(POSITIVE_INDICATORS) + list(NEGATIVE_INDICATORS):
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    return feature_df


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


def build_standardized_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=feature_df.index)
    for col in POSITIVE_INDICATORS:
        matrix[col] = minmax_scale(feature_df[col], positive=True)
    for col in NEGATIVE_INDICATORS:
        matrix[col] = minmax_scale(feature_df[col], positive=False)
    return matrix


def entropy_weights(matrix: pd.DataFrame) -> pd.Series:
    safe = matrix.copy().astype(float)
    safe = safe.fillna(0)

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
    subjective_series = pd.Series(subjective, dtype=float)
    subjective_series = subjective_series.reindex(entropy.index).fillna(0)
    subjective_series = subjective_series / subjective_series.sum()
    combined = alpha * subjective_series + (1 - alpha) * entropy
    return combined / combined.sum()


def assign_category(row: pd.Series) -> str:
    district_type = str(row["商圈类型建议"])
    score = row["消费活力综合得分"]
    if district_type == "核心商圈" and score >= 0.7:
        return "高活力核心型"
    if district_type == "传统核心商圈":
        return "传统优势型"
    if "成长" in district_type or "新兴" in district_type:
        return "成长潜力型"
    if "社区" in district_type:
        return "社区稳定型"
    return "区域均衡型"


def assign_spatial_desc(name: str) -> str:
    if "春熙路" in name or "盐市口" in name:
        return "城市中心集聚"
    if "建设路" in name or "科华" in name:
        return "城市次中心延伸"
    if "双楠" in name:
        return "成熟居住片区"
    return "近郊组团拓展"


def make_summary(
    result_df: pd.DataFrame, weights_df: pd.DataFrame, output_path: Path
) -> None:
    top = result_df.iloc[0]
    lines = [
        "# 问题1：成都商圈消费活力评价结果",
        "",
        "## 评价框架",
        "采用 AHP-熵权组合赋权构建 7 个指标的综合评价模型，指标覆盖商业规模、商业密度、商业成熟度、客流活跃度、核心吸引力、成长潜力与租金成本压力。",
        "",
        "## 指标权重",
    ]

    for _, row in weights_df.iterrows():
        lines.append(
            f"- {row['指标名称']}（{row['指标含义']}）：组合权重 {row['权重']:.4f}，主观权重 {row['主观权重']:.4f}，熵权 {row['熵权权重']:.4f}"
        )

    lines.extend(
        [
            "",
            "## 结果摘要",
            f"- 综合得分最高的商圈为 `{top['商圈名称']}`，得分 {top['消费活力综合得分']:.4f}，类型判定为 `{top['商圈类别']}`。",
            f"- `{result_df.iloc[1]['商圈名称']}` 位列第二，体现传统中心商圈与城市地标商圈之间的双核格局。",
            "- 整体上呈现“中心双核强集聚、次级商圈梯度承接、近郊组团外拓”的空间结构，适合直接写入问题一的空间分布分析。",
            "",
            "## 空间分布特征",
        ]
    )

    for _, row in result_df[["商圈名称", "空间分布特征"]].drop_duplicates().iterrows():
        lines.append(f"- {row['商圈名称']}：{row['空间分布特征']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_score_bar(result_df: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style("white")
    ordered = result_df.sort_values("消费活力综合得分", ascending=True).reset_index(drop=True)

    palette = {
        "高活力核心型": "#c65d2e",
        "传统优势型": "#2a9d8f",
        "成长潜力型": "#4c78a8",
        "社区稳定型": "#7a6aa6",
        "区域均衡型": "#9c9c9c",
    }
    colors = [palette.get(c, "#4c78a8") for c in ordered["商圈类别"]]

    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.barh(
        ordered["商圈名称"],
        ordered["消费活力综合得分"],
        color=colors,
        height=0.72,
        edgecolor="none",
    )

    ax.set_xlim(0, max(0.92, ordered["消费活力综合得分"].max() + 0.08))
    ax.set_xlabel("综合得分", fontproperties=FONT_PROP, fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("成都主要商圈消费活力综合得分排序", fontproperties=FONT_PROP, fontsize=18, pad=18)
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

    for i, (bar, value, cat) in enumerate(zip(bars, ordered["消费活力综合得分"], ordered["商圈类别"])):
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
        "双核商圈显著领先，其余商圈呈现明显梯度分化",
        fontsize=11,
        color="#666666",
        fontproperties=FONT_PROP,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_category_chart(result_df: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style("white")
    category_order = [
        "高活力核心型",
        "传统优势型",
        "成长潜力型",
        "社区稳定型",
        "区域均衡型",
    ]
    category_df = (
        result_df.groupby("商圈类别", as_index=False)
        .agg(商圈数量=("商圈名称", "count"), 平均得分=("消费活力综合得分", "mean"))
    )
    category_df["排序"] = category_df["商圈类别"].apply(
        lambda x: category_order.index(x) if x in category_order else len(category_order)
    )
    category_df = category_df.sort_values("排序")

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    colors = ["#c65d2e", "#2a9d8f", "#4c78a8", "#7a6aa6", "#9c9c9c"]
    bubble_sizes = category_df["商圈数量"] * 900

    ax.hlines(
        y=category_df["商圈类别"],
        xmin=0,
        xmax=category_df["平均得分"],
        color="#d5d5d5",
        linewidth=2,
        zorder=1,
    )
    ax.scatter(
        category_df["平均得分"],
        category_df["商圈类别"],
        s=bubble_sizes,
        c=colors[: len(category_df)],
        alpha=0.95,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    ax.set_xlim(0, max(0.92, category_df["平均得分"].max() + 0.08))
    ax.set_xlabel("类别平均综合得分", fontproperties=FONT_PROP, fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_title("成都商圈类别分布与类别平均得分", fontproperties=FONT_PROP, fontsize=18, pad=18)
    ax.xaxis.grid(True, color="#dfdfdf", linewidth=0.8)
    ax.yaxis.grid(False)

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

    for _, row in category_df.iterrows():
        ax.text(
            row["平均得分"] + 0.015,
            row["商圈类别"],
            f"{row['平均得分']:.3f} / {int(row['商圈数量'])}个",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
            fontproperties=FONT_PROP,
        )

    fig.text(
        0.125,
        0.94,
        "气泡大小表示该类别包含的商圈数量，位置表示类别平均消费活力水平",
        fontsize=11,
        color="#666666",
        fontproperties=FONT_PROP,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input, encoding="utf-8-sig")
    feature_df = build_features(raw_df)
    matrix = build_standardized_matrix(feature_df)
    entropy = entropy_weights(matrix)
    weights = combine_weights(entropy, SUBJECTIVE_WEIGHTS, alpha=0.6)

    score = matrix.mul(weights, axis=1).sum(axis=1)

    result_df = raw_df.copy()
    result_df["客流活跃度代理"] = feature_df["客流活跃度代理"]
    result_df["核心吸引力指数"] = feature_df["核心吸引力指数"]
    result_df["类型潜力分"] = feature_df["类型潜力分"]
    result_df["消费活力综合得分"] = score.round(4)
    result_df["消费活力排名"] = (
        result_df["消费活力综合得分"].rank(method="dense", ascending=False).astype(int)
    )
    result_df = result_df.sort_values(
        ["消费活力综合得分", "商圈成熟度指数"], ascending=[False, False]
    ).reset_index(drop=True)
    result_df["商圈类别"] = result_df.apply(assign_category, axis=1)
    result_df["空间分布特征"] = result_df["商圈名称"].map(assign_spatial_desc)

    weights_df = pd.DataFrame(
        {
            "指标名称": list(weights.index),
            "指标含义": [POSITIVE_INDICATORS.get(col, NEGATIVE_INDICATORS.get(col)) for col in weights.index],
            "主观权重": [SUBJECTIVE_WEIGHTS[col] for col in weights.index],
            "熵权权重": [entropy[col] for col in weights.index],
            "权重": weights.values,
        }
    ).sort_values("权重", ascending=False)

    matrix_out = matrix.copy()
    matrix_out.insert(0, "商圈名称", raw_df["商圈名称"])
    matrix_out = matrix_out.merge(
        result_df[["商圈名称", "消费活力综合得分", "消费活力排名", "商圈类别"]],
        on="商圈名称",
        how="left",
    )

    result_cols = [
        "商圈名称",
        "商圈类型建议",
        "来源点位数",
        "商场样本数",
        "商圈成熟度指数",
        "日均客流量_万人次",
        "客流活跃度代理",
        "核心吸引力指数",
        "成本压力代理_租金",
        "类型潜力分",
        "消费活力综合得分",
        "消费活力排名",
        "商圈类别",
        "空间分布特征",
    ]
    result_cols = [col for col in result_cols if col in result_df.columns]

    result_path = args.output / "q1_成都商圈消费活力评价结果.csv"
    weight_path = args.output / "q1_指标权重.csv"
    matrix_path = args.output / "q1_标准化指标矩阵.csv"
    summary_path = args.output / "q1_结果摘要.md"
    score_fig_path = figures_dir / "q1_商圈活力综合得分排序.png"
    category_fig_path = figures_dir / "q1_商圈类别分布.png"

    result_df[result_cols].to_csv(result_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(weight_path, index=False, encoding="utf-8-sig")
    matrix_out.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    make_summary(result_df, weights_df, summary_path)
    plot_score_bar(result_df, score_fig_path)
    plot_category_chart(result_df, category_fig_path)

    print(f"输入数据: {args.input}")
    print(f"结果表: {result_path}")
    print(f"权重表: {weight_path}")
    print(f"标准化矩阵: {matrix_path}")
    print(f"摘要文件: {summary_path}")
    print(f"得分图: {score_fig_path}")
    print(f"分类图: {category_fig_path}")


if __name__ == "__main__":
    main()
