from sklearn.model_selection import GridSearchCV

def tune_model(pipe, param_grid, cv):

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    return grid