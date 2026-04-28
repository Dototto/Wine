import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataHandler:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_wine_data(self):
        wine = load_wine()

        self.data = pd.DataFrame(wine.data, columns=wine.feature_names)
        self.data["target"] = wine.target

        self.X = self.data.drop(columns=["target", "total_phenols"])
        self.y = self.data["target"]

        return self.data

    def show_info(self):
        print("Prime righe del dataset:")
        print(self.data.head())

        print("\nInformazioni dataset:")
        print(self.data.info())

        print("\nValori mancanti:")
        print(self.data.isnull().sum())

    def clean_data(self):
        # Rimozione duplicati
        print("duplicati", self.data.duplicated().sum())
        #self.data = self.data.drop_duplicates()

        # Gestione valori mancanti
        #self.data = self.data.fillna(self.data.mean(numeric_only=True))

        #self.X = self.data.drop("target", axis=1)
        #self.y = self.data["target"]

        return self.data

    def plot_correlation_heatmap(self, figsize=(10, 8), threshold = 0.75, annot=False, cmap="coolwarm"):
        if self.data is None:
            raise ValueError("Carica prima i dati!")

        corr_matrix = self.data.corr(numeric_only=True)

        # Maschera per mostrare solo correlazioni sopra soglia
        mask = np.abs(corr_matrix) < threshold

        plt.figure(figsize=figsize)

        sns.heatmap(
            corr_matrix,
            mask=mask,  # NASCONDE valori sotto soglia
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            annot_kws={"size": 8}
        )

        plt.title(f"Heatmap correlazioni (|corr| ≥ {threshold})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_test

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test