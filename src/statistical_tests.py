import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score
from src.pipelines import create_pipeline


def get_cv_scores(model, X, y, cv):

    pipe = create_pipeline(
        model,
        resampler=None,
        use_scaler=True
    )

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    return scores


def compare_models(model1, model2, X, y, cv,
                   name1="RF",
                   name2="CatBoost"):

    scores1 = get_cv_scores(model1, X, y, cv)
    scores2 = get_cv_scores(model2, X, y, cv)

    print(f"\n{name1} scores:")
    print(scores1)

    print(f"\n{name2} scores:")
    print(scores2)


    print("\n===== SHAPIRO-WILK =====")

    sh1 = stats.shapiro(scores1)
    sh2 = stats.shapiro(scores2)

    print(f"{name1}: stat={sh1.statistic:.4f}, p={sh1.pvalue:.4f}")
    print(f"{name2}: stat={sh2.statistic:.4f}, p={sh2.pvalue:.4f}")

    if sh1.pvalue < 0.05 or sh2.pvalue < 0.05:
        print("Należy odrzucić hipotezę zerową")
    else:
        print("Nie ma podstaw do odrzucenia hipotezy zerowej")


    print("\n===== MANN-WHITNEY U =====")

    mw = stats.mannwhitneyu(
        scores1,
        scores2,
        alternative="two-sided"
    )

    print(f"stat={mw.statistic:.4f}, p={mw.pvalue:.6f}")

    if mw.pvalue < 0.05:
        print("Należy odrzucić hipotezę zerową")
    else:
        print("Nie ma podstaw do odrzucenia hipotezy zerowej")


    print("\n===== PAIRED T-TEST =====")

    ttest = stats.ttest_rel(scores1, scores2)

    print(f"stat={ttest.statistic:.4f}, p={ttest.pvalue:.6f}")

    if ttest.pvalue < 0.05:
        print("Należy odrzucić hipotezę zerową")
    else:
        print("Nie ma podstaw do odrzucenia hipotezy zerowej")



    print("\n===== WILCOXON =====")

    wil = stats.wilcoxon(scores1, scores2)

    print(f"stat={wil.statistic:.4f}, p={wil.pvalue:.6f}")

    if wil.pvalue < 0.05:
        print("Należy odrzucić hipotezę zerową")
    else:
        print("Nie ma podstaw do odrzucenia hipotezy zerowej")

    return pd.DataFrame({
        name1: scores1,
        name2: scores2
    })

