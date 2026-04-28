import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataHandler:
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__scaler = StandardScaler()

    def load_wine_data(self):
        wine = load_wine()

        self.__data = pd.DataFrame(wine.data, columns=wine.feature_names)
        self.__data["target"] = wine.target

        self.__X = self.__data.drop(columns=["target", "total_phenols"])
        self.__y = self.__data["target"]


    def show_info(self):
        print("Prime righe del dataset:")
        print(self.__data.head())

        print("\nInformazioni dataset:")
        print(self.__data.info())

        print("\nValori mancanti:")
        print(self.__data.isnull().sum())

    def clean_data(self):
        # Rimozione duplicati
        print("duplicati", self.__data.duplicated().sum())


    def plot_correlation_heatmap(self, figsize=(10, 8), threshold = 0.75, annot=False, cmap="coolwarm"):
        if self.__data is None:
            raise ValueError("Carica prima i dati!")

        corr_matrix = self.__data.corr(numeric_only=True)

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
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__X,
            self.__y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.__y
        )

        return self.__X_train, self.__X_test, self.__y_train, self.__y_test

    def scale_data(self):
        self.__X_train = self.__scaler.fit_transform(self.__X_train)
        self.__X_test = self.__scaler.transform(self.__X_test)

    def get_train_test_data(self):
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test

    def get_X(self):
        return self.__X

    def get_y(self):
        return self.__y