import json
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)

from src.pipelines import create_pipeline
from src.config import PARAM_GRIDS, MODELS
from src.data_loader import load_data
from src.preprocessing import prepare_data


def nested_cv_evaluation(model_name, X, y):

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    pipe = create_pipeline(
        model=MODELS[model_name],
        resampler=None,
        use_scaler=False
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRIDS[model_name],
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    scores = cross_validate(
        estimator=grid,
        X=X,
        y=y,
        cv=outer_cv,
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=False
    )

    mean_score = scores["test_score"].mean()
    std_score = scores["test_score"].std()

    return mean_score, std_score


def find_best_params_on_full_dataset(model_name, X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    pipe = create_pipeline(
        model=MODELS[model_name],
        resampler=None,
        use_scaler=False
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRIDS[model_name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    grid.fit(X, y)

    return (
        grid.best_params_,
        grid.best_score_,
        grid.best_estimator_
    )


# MAIN
df = load_data("Dataset/WineQT.csv")

mode = "binary"

X, y = prepare_data(df, mode=mode)

results = []

for model_name in PARAM_GRIDS:

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    # 1. Uczciwa ocena jakości modelu
    mean_score, std_score = nested_cv_evaluation(model_name,X,y)

    print(f"Nested CV F1-macro: {mean_score:.4f} ({std_score:.4f})")

    # 2. Znalezienie najlepszych parametrów na całym zbiorze
    best_params, best_cv_score, best_model = find_best_params_on_full_dataset(model_name,X,y)

    print("Best params:")
    print(best_params)

    results.append({
        "model": model_name,
        "nested_cv_mean_f1": round(mean_score, 4),
        "nested_cv_std_f1": round(std_score, 4),
        "best_gridsearch_cv_score": round(best_cv_score, 4),
        "best_params": best_params
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="nested_cv_mean_f1",
    ascending=False
)

print("\n")
print(results_df)

results_df.to_json(
    "nested_autoML_results.json",
    orient="records",
    indent=4
)