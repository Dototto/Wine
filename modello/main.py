import os

import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from data_handler import DataHandler

def plot_confusion_matrix_cv(model, X, y, title, filename, cv=5, show=False):
    os.makedirs("risultati", exist_ok=True)

    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred_cv)

    class_names = ["class_0", "class_1", "class_2"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    fig.tight_layout()

    path = f"risultati/{filename}"
    fig.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, title, filename, show=False):
    os.makedirs("risultati", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["class_0", "class_1", "class_2"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    fig.tight_layout()

    path = f"risultati/{filename}"
    fig.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

def run_cross_validation(X, y, cv=5):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    plot_confusion_matrix_cv(
        pipeline,
        X,
        y,
        "Confusion Matrix CV - Logistic Regression",
        "cm_cv_lr.png",
        cv=cv
    )

    print("\n=== CROSS VALIDATION LOGISTIC REGRESSION ===")

    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"\n{metric.upper()}:")
        print(f"  valori: {scores}")
        print(f"  media: {scores.mean():.4f}")
        print(f"  std:   {scores.std():.4f}")

def run_cross_validation_knn(X, y, cv=5):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=11))
    ])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    plot_confusion_matrix_cv(
        pipeline,
        X,
        y,
        "Confusion Matrix CV - KNN",
        "cm_cv_knn.png",
        cv=cv
    )

    print("\n=== CROSS VALIDATION KNN ===")

    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"\n{metric.upper()}:")
        print(f"  valori: {scores}")
        print(f"  media: {scores.mean():.4f}")
        print(f"  std:   {scores.std():.4f}")

def main():
    # Inizializzazione
    data_handler = DataHandler()

    # Caricamento e ispezione
    data_handler.load_wine_data()
    data_handler.show_info()

    # Pulizia e preprocessing
    data_handler.drop_data()
    data_handler.plot_correlation_heatmap()
    data_handler.split_data(test_size=0.2)
    data_handler.scale_data()

    # Dati pronti
    X_train, X_test, y_train, y_test = data_handler.get_train_test_data()

    # Modello
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)


    # Predizione
    training_data_prediction = model.predict(X_train)
    test_data_prediction = model.predict(X_test)

    # Valutazione
    print(f"TRAIN Regressione Logistica:\n{classification_report(y_train, training_data_prediction)}")
    print(f"TEST Regressione Logistica:\n{classification_report(y_test, test_data_prediction)}")

    # Cross-validation su tutto il dataset
    run_cross_validation(data_handler.get_X(), data_handler.get_y(), cv=5)

    plot_confusion_matrix(
        y_test,
        test_data_prediction,
        "Confusion Matrix - Logistic Regression",
        "cm_lr.png"
    )

    ##########
    modelKNN = KNeighborsClassifier(n_neighbors=11)
    modelKNN.fit(X_train, y_train)

    training_data_predictionKNN = modelKNN.predict(X_train)
    test_data_predictionKNN = modelKNN.predict(X_test)

    print(f"TRAIN KNN:\n{classification_report(y_train, training_data_predictionKNN)}")
    print(f"TEST KNN:\n{classification_report(y_test, test_data_predictionKNN)}")

    run_cross_validation_knn(data_handler.get_X(), data_handler.get_y(), cv=5)

    plot_confusion_matrix(
        y_test,
        test_data_predictionKNN,
        "Confusion Matrix - KNN",
        "cm_knn.png"
    )

    joblib.dump(model, "wine_model.pkl")
    joblib.dump(modelKNN, "wine_modelKNN.pkl")
    joblib.dump(data_handler.get_X().columns.tolist(), "model_columns.pkl")
    joblib.dump(data_handler.get_scaler(), 'scaler.pkl')



if __name__ == "__main__":
    main()