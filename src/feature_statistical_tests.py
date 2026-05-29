import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def compare_feature_between_classes(
        df,
        feature,
        class_column="quality",
        class_a=5,
        class_b=6
):

    # dane grup
    x = df[df[class_column] == class_a][feature]
    y = df[df[class_column] == class_b][feature]

    print(f"\n===== FEATURE: {feature} =====")

    print(f"\nClass {class_a}")
    print(x.describe())

    print(f"\nClass {class_b}")
    print(y.describe())

    # =========================================
    # SHAPIRO
    # =========================================

    print("\n===== SHAPIRO-WILK =====")

    shapiro_x = stats.shapiro(x)
    shapiro_y = stats.shapiro(y)

    print(
        f"Class {class_a}: "
        f"stat={shapiro_x.statistic:.4f}, "
        f"p={shapiro_x.pvalue:.6f}"
    )

    print(
        f"Class {class_b}: "
        f"stat={shapiro_y.statistic:.4f}, "
        f"p={shapiro_y.pvalue:.6f}"
    )

    # =========================================
    # LEVENE
    # =========================================

    print("\n===== LEVENE =====")

    lev = stats.levene(x, y)

    print(
        f"stat={lev.statistic:.4f}, "
        f"p={lev.pvalue:.6f}"
    )

    # =========================================
    # T-TEST
    # =========================================

    print("\n===== WELCH T-TEST =====")

    ttest = stats.ttest_ind(
        x,
        y,
        equal_var=False
    )

    print(
        f"stat={ttest.statistic:.4f}, "
        f"p={ttest.pvalue:.6f}"
    )

    # =========================================
    # MANN-WHITNEY
    # =========================================

    print("\n===== MANN-WHITNEY U =====")

    mw = stats.mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    print(
        f"stat={mw.statistic:.4f}, "
        f"p={mw.pvalue:.6f}"
    )

    # =========================================
    # CI
    # =========================================

    mean_diff = x.mean() - y.mean()

    se = (
        (x.var(ddof=1) / len(x))
        +
        (y.var(ddof=1) / len(y))
    ) ** 0.5

    ci_low, ci_high = stats.t.interval(
        0.95,
        df=len(x) + len(y) - 2,
        loc=mean_diff,
        scale=se
    )

    print("\n===== CONFIDENCE INTERVAL =====")

    print(f"Mean difference: {mean_diff:.4f}")

    print(
        f"95% CI: "
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )

    # =========================================
    # WYKRES
    # =========================================

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=class_column,
        y=feature
    )

    sns.stripplot(
        data=df,
        x=class_column,
        y=feature,
        color="black",
        alpha=0.3
    )

    plt.title(f"{feature} vs quality")
    plt.grid()
    plt.tight_layout()

    plt.show()


