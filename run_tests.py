from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.config import MODELS
from sklearn.model_selection import StratifiedKFold
from src.statistical_tests import compare_models


df = load_data("Dataset/WineQT.csv")

mode = "binary"

X, y = prepare_data(df, mode)

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

rf = MODELS["RandomForestClassifier"]
cat = MODELS["CatBoost"]

results = compare_models(
    rf,
    cat,
    X,
    y,
    cv,
    name1="RandomForest",
    name2="CatBoost"
)

print(results)

