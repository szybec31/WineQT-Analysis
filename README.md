# Statistical data analysis - laboratories

## Project Overview
This project is an end-to-end Data Science study based on the **WineQT dataset**. The main goal is to analyze the 
chemical properties of wine and predict its overall quality score. 

Instead of just training a single model, this project covers the full machine learning workflow:
- **Exploratory Data Analysis (EDA):** Finding patterns and correlations in wine chemistry.
- **Statistical Testing:** Checking if the differences between features and models are actually meaningful.
- **AutoML (Automated ML):** Using a custom **Nested AutoML** pipeline to automatically find the best models and fine-tune their parameters.
- **Fairness:** Checking if the models make fair and unbiased predictions across different groups of data.

## Dataset Description

Ready dataset has been taken from Kaggle.com ([WineQT_dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)).

This data frame contains the following columns:
- 1 - fixed acidity
- 2 - volatile acidity
- 3 - citric acid
- 4 - residual sugar
- 5 - chlorides
- 6 - free sulfur dioxide
- 7 - total sulfur dioxide
- 8 - density
- 9 - pH
- 10 - sulphates
- 11 - alcohol
\
Output variable (based on sensory data):
- 12 - quality (score between 3 and 8)

## Download and run the project:
1. Clone the repository:
   ```
    git clone https://github.com/szybec31/WineQT-Analysis.git
    cd WineQT-Analysis
   ```
2. Create a virtual environment:
    ```
    python -m venv myenv
    myenv\Scripts\activate       # Windows
    source myenv/bin/activate    # Linux/Mac 
    ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Run notebook (optional):
   ```
   python -m jupyterlab
   ```
## Important files and directories description:

- `charts/` - Directory where generated plots and visualizations are saved
- `Dataset/WineQT.csv` - CSV file containing the wine quality dataset
- `notebooks/` - Jupyter notebooks for specific laboratory assignments
    - `01_eda.ipynb` - Exploratory Data Analysis (EDA) of the WineQT dataset
    - `04_regression.ipynb` - Development, training, and evaluation of regression models
    - `05_statistcal_tests_models.ipynb` - Statistical testing for model performance comparison
    - `05_statistcal_tests_features.ipynb` - Statistical tests concerning feature significance and relationships
    - `06_fairness.ipynb` - Assessment of model fairness and bias mitigation analysis
- `src/` - Core source folder containing helper modules and business logic
    - `config.py` - Global configuration settings, file paths, and hyperparameter definitions
    - `data_loader.py` - Script responsible for loading, partitioning, and initializing the dataset
    - `eda.py` - Utility functions for generating descriptive statistics and exploratory charts
    - `evaluation.py` - Model evaluation metrics (e.g., RMSE, MAE, $R^2$) and validation functions
    - `feature_statistcal_tests.py` - Implementation of statistical tests to analyze feature-to-feature relationships
    - `models.py` - Definitions and architectures of the machine learning models used
    - `pipelines.py` - Definition of data pipelines combining preprocessing steps with model training
    - `preprocessing.py` - Data cleaning, missing value handling, scaling, and feature engineering
    - `regression.py` - Logic handling the training, fitting, and tuning of regression models
    - `results.py` - Functions for formatting, logging, and saving experiment outcomes
    - `statistcal_tests.py` - General implementations of statistical tests utilized across the project
- `main.py` - Main entry point to run the standard training and evaluation pipeline
- `automl.py` - Alternative entry point executing the automated machine learning hyperparameter tuning pipeline
- `nested_autoML_results.json` - JSON file storing the results, metrics, and configurations from the Nested AutoML execution
- `requirements.txt` - Configuration file listing all dependencies required to run the project
- `.gitignore` - Specifies intentionally untracked files and directories that Git should ignore

## Authors:
- Szymon Bęczkowski
- Piotr Kontny
- Kamil Marczyński