from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import optuna
from sklearn.model_selection import cross_val_score
from src.pipelines import create_pipeline


# To jest obecne rozwiązanie, które może nie być najlepsze i być może się zmieni. Generalnie za pomocą biblioteki
# Optuna poniżej wybrane 3 parametry dla modeli GradientBoosting i CatBoost są losowane jako różne kombinacje
# n_trials-razy, czyli tak zwane hyperparameter tuning. Obecne rozwiązanie jest bardziej żeby pokazać jak i że ta
# funkcja z AutoML działa. Do ostatecznej prezentacji można ustawić liczbę trialów na znacznie większą wartość, żeby
# mniej więcej ustalić, która kombinacja byłaby najlepsza

def optimize_model(model_name, model, X, y, cv):

    def objective(trial):

        if model_name == "GradientBoostingClassifier":
            params = {
                # liczba drzew, im więcej tym większa szansa na overfitting
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                # szybkość uczenia się - 0,1 szybciej; 0,01 wolniej, ale dokładniej, jeśli
                # learning_rate jest za długi, to nie dojdzie do idealnej odpowiedzi, gdzie "Loss" jest najmniejszy i predykcja
                # jest najlepsza
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                # głębokość drzewa (ilość liści powiedzmy)
                "max_depth": trial.suggest_int("max_depth", 3, 8),
            }

            model = GradientBoostingClassifier(**params, random_state=42)

        elif model_name == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 400), # to samo co n_estimators tyle że dla CatBoost
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "depth": trial.suggest_int("depth", 4, 10), # to samo co max_depth
            }

            model = CatBoostClassifier(**params, verbose=0, random_state=42)

        else:
            return 0

        pipe = create_pipeline(model, resampler=None, use_scaler=True)

        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=cv,
            scoring="f1_macro"
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    # Liczbę prób można zmieniać, ale wstępnie do testów dałem tylko 10, bo długo się mieli
    study.optimize(objective, n_trials=10)

    return study.best_params, study.best_value