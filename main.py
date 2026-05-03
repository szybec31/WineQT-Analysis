from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.config import MODELS, SCORING, RESAMPLERS
from src.automl import optimize_model
from src.pipelines import create_pipeline
from src.evaluation import evaluate_model, mean_confusion_matrix, plot_roc_curves
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
#from imblearn.over_sampling import SMOTE
import seaborn as sns

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
X, y = prepare_data(df,mode = mode)

# CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

for res_name, resampler in RESAMPLERS.items():
    for name, model in MODELS.items():

        print(f"\n===== {name} | {res_name} =====")

        pipe = create_pipeline(model,resampler=resampler,use_scaler=True)

        scores = evaluate_model(pipe, X, y, skf, SCORING)

        print(scores)

        results.append({
            "model": name,
            "resampling": res_name,
            **scores
        })

results_df = pd.DataFrame(results)

# jeśli masz stringi
results_df["f1"] = results_df["f1"].str.split().str[0].astype(float)

sns.barplot(
    data=results_df,
    x="resampling",
    y="f1",
    hue="model"
)

plt.title("Resampling comparison (F1)")
plt.tight_layout()
plt.show()

# Macierz pomyłek dla każdego modelu

resampling = "none" # Wykres tylko dla danego trybu resamplingu

models = []

for res_name, resampler in RESAMPLERS.items():
    if res_name != resampling:
        continue

    for name, model in MODELS.items():
        pipe = create_pipeline(model, resampler=resampler, use_scaler=True)

        mean_cm, labels = mean_confusion_matrix(pipe, X, y, skf)

        models.append((name, mean_cm, labels))

n_models = len(models)

# Nie wiem ile będzie modelów w przszłości także to ma znaleźć najbardziej optymalną konfigurację subplotu
cols = 2
rows = (n_models + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
axes = axes.flatten()

for i, (name, cm, labels) in enumerate(models):
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[i]
    )
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

# Na obecny moment jest 5 modeli dlatego przy subplocie 3 x 2 pozostaje jeden pusty także go usuwam
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f"Mean Confusion Matrices ({resampling})", fontsize=16)
plt.tight_layout()
plt.show()


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


for name, model in MODELS.items():

    if name not in ["GradientBoostingClassifier", "CatBoost"]:
        continue

    print(f"\n===== AutoML: {name} =====")

    best_params, best_score = optimize_model(name, model, X, y, skf)

    print("Best params:", best_params)
    print("Best score:", best_score)
