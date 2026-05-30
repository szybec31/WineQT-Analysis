import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA():
    def __init__(self,df,show=False):
        self.df = df
        self.show = show
        self.X = None
        self.y = None

    def basic_info(self):
        print(self.df.head(10))
        print(self.df.info())
        print(self.df.describe())
        print(self.df.isnull().sum())

    def description(self):
        self.df = self.df.drop(['Id'], axis=1)
        self.y = self.df['quality']
        self.X = self.df.drop(['quality'], axis=1)
        categorial_features = self.X.select_dtypes(include=['category', 'str', 'object']).columns.tolist()
        numerical_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        n_samples = len(self.X)
        n_features = len(self.X.T)
        classes, counts = np.unique(self.y, return_counts=True)
        imbalanced_ratio = np.max(counts) / np.min(counts)

        print("n_samples: ", n_samples)
        print("n_features: ", n_features)
        print("categorical: ", len(categorial_features))
        print("numerical: ", len(numerical_features))
        print("classes: ", classes)
        print("class samples: ", counts)
        print("imbalanced ratio: ", imbalanced_ratio)

    def class_distribution(self):
        sns.countplot(x="quality", data=self.df)
        plt.title("Class distribution")
        if self.show:
            plt.show()
        plt.savefig("charts/Class_distribution.png")

    def outliners(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.X)
        plt.xticks(rotation=45)
        plt.title("Boxplot - outliers detection")
        if self.show:
            plt.show()
        plt.savefig("charts/Boxplot_outliers_detection.png")

    def correlation(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        if self.show:
            plt.show()
        plt.savefig("charts/Correlation_Matrix.png")

    def delete_outliers(self,X,y):
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        labels = iso.fit_predict(X)

        mask = labels == 1

        X = X[mask]
        y = y[mask]

        print(f"Usunięto {(labels==-1).sum()} rekordów")
        idx = X["total sulfur dioxide"].nlargest(2).index

        X = X.drop(idx)
        y = y.drop(idx)

        return X,y