# src/regression.py

from sklearn.model_selection import cross_validate

def evaluate_regression(model, X, y, cv):

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    results = {
        "MAE": f"{-scores['test_mae'].mean():.3f} ({scores['test_mae'].std():.3f})",
        "RMSE": f"{-scores['test_rmse'].mean():.3f} ({scores['test_rmse'].std():.3f})",
        "R2": f"{scores['test_r2'].mean():.3f} ({scores['test_r2'].std():.3f})"
    }

    return results