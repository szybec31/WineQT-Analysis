from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from src.pipelines import create_pipeline
from src.config import PARAM_GRIDS
from src.config import MODELS, SCORING, RESAMPLERS
from src.data_loader import load_data
from src.preprocessing import prepare_data
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


def grid_search_models(model_name, X, y, cv, param_grid):

    best_score = 0
    best_params = None

    all_params = list(ParameterGrid(param_grid))

    total = len(all_params)

    for i, params in enumerate(all_params):

        print(f"\n[{i+1}/{total}] Testing combination:")

        print(params)

        if model_name == "GradientBoostingClassifier":

            model = GradientBoostingClassifier(
                n_estimators=params["model__n_estimators"],
                learning_rate=params["model__learning_rate"],
                max_depth=params["model__max_depth"],
                random_state=42
            )

        elif model_name == "CatBoost":

            model = CatBoostClassifier(
                iterations=params["model__iterations"],
                learning_rate=params["model__learning_rate"],
                depth=params["model__depth"],
                verbose=0,
                random_state=42
            )

        else:
            continue

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

        mean_score = scores.mean()

        print(f"Score: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


df = load_data("Dataset/WineQT.csv")

# Mod "basic" dla wszystkich 6 klas, "4multiclass" dla 4 klas i "binary" dla dwóch klas
mode = "binary"

# preprocess
X, y = prepare_data(df,mode = mode)

# CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []


for name, model in MODELS.items():

    if name not in PARAM_GRIDS:
        continue

    print(f"\n===== GRID SEARCH: {name} =====")

    best_params, best_score = grid_search_models(
        name,
        X,
        y,
        skf,
        PARAM_GRIDS[name]
    )

    print("\n===== FINAL RESULT =====")
    print("Best params:", best_params)
    print("Best score:", best_score)

