
from __future__ import annotations

import warnings
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

# 解决中文标题/标签乱码问题：设置中文字体和负号显示
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

def load_data(path: str) -> pd.DataFrame:
    """读取数据并清理无用索引列。"""
    df = pd.read_csv(path)
    # 删除 pandas 自动保存的索引列
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def build_preprocessor(df: pd.DataFrame, target_cols: list[str]) -> ColumnTransformer:
    """构建数值/类别列的预处理流程。"""
    feature_cols = [c for c in df.columns if c not in target_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,  # 兼容新版 scikit-learn 的稀疏输出参数
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def classification_task(df: pd.DataFrame) -> None:
    """基于 V22 做二分类预测。"""
    target = "V22"
    if target not in df.columns:
        print(f"未找到分类目标列 {target}")
        return

    # 为兼顾性能，随机抽样最多 200k 行进行训练
    sample_df = df.sample(n=min(200_000, len(df)), random_state=42)

    preprocessor = build_preprocessor(sample_df, [target])
    X = sample_df.drop(columns=[target])
    y = sample_df[target].fillna(sample_df[target].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=200, n_jobs=1)),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("=== 分类任务: 预测 V22 ===")
    # 混淆矩阵（文本 + 图）
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵（行=真实标签, 列=预测标签）:")
    print(cm)
    os.makedirs("figs", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("V22 混淆矩阵")
    plt.tight_layout()
    plt.savefig("figs/v22_confusion_matrix.png")
    plt.close()
    print(classification_report(y_test, y_pred, digits=3))


def regression_task(df: pd.DataFrame) -> None:
    """使用线性回归预测 V20（连续变量）。"""
    target = "V20"
    if target not in df.columns:
        print(f"未找到回归目标列 {target}")
        return

    sample_df = df.sample(n=min(200_000, len(df)), random_state=42).copy()
    # 将目标列转换为数值，无法转换的标记为 NaN 并剔除
    sample_df[target] = pd.to_numeric(sample_df[target], errors="coerce")
    sample_df = sample_df.dropna(subset=[target])

    preprocessor = build_preprocessor(sample_df, [target])
    X = sample_df.drop(columns=[target])
    y = sample_df[target].fillna(sample_df[target].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("=== 线性回归任务: 预测 V20 ===")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R2 : {r2_score(y_test, y_pred):.4f}")


def association_task(df: pd.DataFrame) -> None:
    """对主要的类别字段进行关联规则挖掘。"""
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not cat_cols:
        print("没有可用的类别字段用于关联分析")
        return

    sample = df[cat_cols].sample(n=min(5000, len(df)), random_state=42)
    # 组合每行的非空取值作为事务
    transactions = (
        sample.fillna("缺失")
        .astype(str)
        .apply(lambda row: [f"{col}={val}" for col, val in row.items()], axis=1)
    )
    # 手动构造 one-hot 编码矩阵
    all_items = sorted({item for tx in transactions for item in tx})
    encoded = pd.DataFrame(False, index=transactions.index, columns=all_items)
    for idx, tx in transactions.items():
        encoded.loc[idx, tx] = True

    frequent = apriori(encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1.1)
    rules = rules.sort_values(by="lift", ascending=False).head(10)

    print("=== 关联规则（前10条） ===")
    if rules.empty:
        print("未挖掘到满足阈值的规则，请降低 min_support 或 min_threshold。")
        return
    for _, row in rules.iterrows():
        antecedents = ", ".join(row["antecedents"])
        consequents = ", ".join(row["consequents"])
        print(
            f"{antecedents} -> {consequents} | support={row['support']:.3f}, "
            f"confidence={row['confidence']:.3f}, lift={row['lift']:.3f}"
        )

    #可视化部分------

def visualization_task(df: pd.DataFrame) -> None:
    """生成不少于4种图形，保存到 figs 目录。"""
    os.makedirs("figs", exist_ok=True)
    sns.set_theme(style="whitegrid")
    sample = df.sample(n=min(10000, len(df)), random_state=42).copy()

    # 1. 目标列分布（分类）
    plt.figure(figsize=(6, 4))
    sns.countplot(x="V22", data=sample)
    plt.title("V22 目标分布")
    plt.tight_layout()
    plt.savefig("figs/v22_count.png")
    plt.close()

    # 2. 连续变量直方图
    if "V20" in sample.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(sample["V20"].dropna(), bins=50, kde=True)
        plt.title("V20 分布直方图")
        plt.tight_layout()
        plt.savefig("figs/v20_hist.png")
        plt.close()

    # 3. 箱线图（按目标分组的 V19）
    if "V19" in sample.columns and "V22" in sample.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="V22", y="V19", data=sample, showfliers=False)
        plt.title("V19 vs V22 箱线图")
        plt.tight_layout()
        plt.savefig("figs/v19_box.png")
        plt.close()

    # 4. 数值特征相关系数热力图
    numeric_cols = [c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])]
    if len(numeric_cols) >= 2:
        corr = sample[numeric_cols].corr().fillna(0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("数值特征相关性热力图")
        plt.tight_layout()
        plt.savefig("figs/corr_heatmap.png")
        plt.close()

    # 5. 类别特征 Top10 频次条形图（示例 V14）
    if "V14" in sample.columns:
        plt.figure(figsize=(6, 4))
        top_counts = sample["V14"].fillna("缺失").value_counts().head(10)
        sns.barplot(x=top_counts.values, y=top_counts.index, orient="h")
        plt.title("V14 Top10 频次")
        plt.tight_layout()
        plt.savefig("figs/v14_top10.png")
        plt.close()

    print("图表已保存到 figs 目录：v22_count.png, v20_hist.png, v19_box.png, corr_heatmap.png, v14_top10.png")


def main():
    path = "data/case_data.csv"
    print("加载数据中...")
    df = load_data(path)
    print(f"数据规模: {df.shape[0]} 行, {df.shape[1]} 列")

    classification_task(df)
    print()
    regression_task(df)
    print()
    association_task(df)
    print()
    visualization_task(df)


if __name__ == "__main__":
    main()

