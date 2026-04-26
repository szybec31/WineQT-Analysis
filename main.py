from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.config import MODELS, SCORING
from src.pipelines import create_pipeline
from src.evaluation import evaluate_model
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit
)

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

# load
df = load_data("Dataset/WineQT.csv")

# preprocess
X, y = prepare_data(df)

y = (y == 6).astype(int)

# CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in sss.split(X, y):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

smote = SMOTE(k_neighbors=2, random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# przed SMOTE
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title("Train set BEFORE SMOTE")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

# po SMOTE
sns.countplot(x=y_train_smote, ax=axes[1])
axes[1].set_title("Train set AFTER SMOTE")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

for name, model in MODELS.items():

    print(f"\n===== {name} =====")

    pipe = create_pipeline(model,True,False)

    scores = evaluate_model(pipe, X, y, skf, SCORING)

    print(scores)

    results.append({
        "model": name,
        **scores
    })

    pipe.fit(X_train, y_train)

    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.3f})"
        )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(results)
print("\n=== SUMMARY ===")
print(results_df)

