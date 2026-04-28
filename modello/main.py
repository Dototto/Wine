from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modello.data_handler import DataHandler

def run_cross_validation(X, y, cv=5):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
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
    #data_handler.clean_data()
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
    run_cross_validation(data_handler.X, data_handler.y, cv=5)
    run_cross_validation_knn(data_handler.X, data_handler.y, cv=5)

    ##########
    modelKNN = KNeighborsClassifier(n_neighbors=11)
    modelKNN.fit(X_train, y_train)

    training_data_predictionKNN = modelKNN.predict(X_train)
    test_data_predictionKNN = modelKNN.predict(X_test)

    print(f"TRAIN KNN:\n{classification_report(y_train, training_data_predictionKNN)}")
    print(f"TEST KNN:\n{classification_report(y_test, test_data_predictionKNN)}")



if __name__ == "__main__":
    main()