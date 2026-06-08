import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from src.eda import EDA
from src.preprocessing import prepare_data
from src.config import MODELS, SCORING, RESAMPLERS, PARAM_GRIDS
from src.pipelines import create_pipeline
from src.evaluation import (evaluate_model, mean_confusion_matrix, plot_roc_curves,
                            plot_rf_resampling_confusion_matrices, RF_feature_importance)
from src.results import create_results_tables, plot_results
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

# =============================================
# 1. Przeprowadzając testy należy wybrać ilość klas jaki chcemy uwzględnić w naszym problemie
# 2. Natępnie wybieramy kroki jakie chcemy uwzględnić w eksperymencie.
#    a. Dostosowujemy pipeline ---> pipe = create_pipeline(model,resampler=None,use_scaler=True)
#    b. lub włączamy/wyłączamy poszczególne elementy w pliku config.py
# =============================================

# load
df = load_data("Dataset/WineQT.csv")

# Mod "basic" dla wszystkich 6 klas, "4multiclass" dla 4 klas i "binary" dla dwóch klas
mode = "binary"

# preprocess
X, y = prepare_data(df, mode = mode)

# Odkomentuj dwie poniższe linie by przeprowadzić eksperyment, który usuwa 5% rekordów z odstającymi wartościami
#eda = EDA(X, y)
#X,y = eda.delete_outliers(X,y)

# CV - (dla wszystkich klas użyć 5 splitów, w pozostałych może być 10)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []
flag = 0


for res_name, resampler in RESAMPLERS.items():
    for name, model in MODELS.items():

        print(f"\n===== {name} | {res_name} =====")
        tree_models = [
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "CatBoost"
        ]
        use_scaler = name not in tree_models
        pipe = create_pipeline(model,resampler=resampler,use_scaler=use_scaler)

        scores = evaluate_model(pipe, X, y, skf, SCORING)

        if name == "RandomForestClassifier" and flag == 0:
            RF_feature_importance(pipe,X,y) # ważność cech według modelu RF
            flag = 1

        print(scores)

        results.append({
            "model": name,
            "resampling": res_name,

            "f1_macro": scores["f1_macro"],
            "bal_acc": scores["bal_acc"],
            "precision": scores["precision"],
            "recall": scores["recall"],

            # do wykresów
            "f1_mean": float(scores["f1_macro"].split()[0]),
            "bal_acc_mean": float(scores["bal_acc"].split()[0]),
        })
results_df = create_results_tables(results)
# jeśli masz stringi
#results_df["f1"] = results_df["f1"].str.split().str[0].astype(float)

plot_results(results_df)

plot_rf_resampling_confusion_matrices(
    X=X,
    y=y,
    cv=skf,
    models_dict=MODELS,
    resamplers_dict=RESAMPLERS,
    create_pipeline=create_pipeline
)


# Krzywa ROC dla problemu binarnego

if mode == "binary":

    resampling = "none"
    models = []

    for res_name, resampler in RESAMPLERS.items():
        if res_name != resampling:
            continue

        for name, model in MODELS.items():
            pipe = create_pipeline(model, resampler=resampler, use_scaler=True)
            models.append((name, pipe))

    plot_roc_curves(models, X, y, skf)


