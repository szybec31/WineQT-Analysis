from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.config import MODELS, SCORING
from src.pipelines import create_pipeline
from src.evaluation import evaluate_model
from src.models import tune_model

from sklearn.model_selection import StratifiedKFold
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

# load
df = load_data("Dataset/WineQT.csv")

# preprocess
X, y = prepare_data(df)

# CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

for name, model in MODELS.items():

    print(f"\n===== {name} =====")

    pipe = create_pipeline(model,True,False)

    scores = evaluate_model(pipe, X, y, skf, SCORING)

    print(scores)

    results.append({
        "model": name,
        **scores
    })

results_df = pd.DataFrame(results)
print("\n=== SUMMARY ===")
print(results_df)