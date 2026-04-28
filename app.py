from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Caricamento modelli
modello_lr = joblib.load("modello/wine_model.pkl")
modello_knn = joblib.load("modello/wine_modelKNN.pkl")
colonne_modello = joblib.load("modello/model_columns.pkl")
scaler_modello = joblib.load("modello/scaler.pkl")


app = FastAPI(title="Wine Prediction API")


class_names = {
    0: "class_0",
    1: "class_1",
    2: "class_2"
}


class DatiWine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.get("/")
def home():
    return {"message": "Wine Prediction API attiva"}


@app.post("/predict")
def predict(data: DatiWine):
    wine = pd.DataFrame([{
        "alcohol": data.alcohol,
        "malic_acid": data.malic_acid,
        "ash": data.ash,
        "alcalinity_of_ash": data.alcalinity_of_ash,
        "magnesium": data.magnesium,
        "flavanoids": data.flavanoids,
        "nonflavanoid_phenols": data.nonflavanoid_phenols,
        "proanthocyanins": data.proanthocyanins,
        "color_intensity": data.color_intensity,
        "hue": data.hue,
        "od280/od315_of_diluted_wines": data.od280_od315_of_diluted_wines,
        "proline": data.proline
    }])

    wine = wine[colonne_modello]
    wine_scaled = scaler_modello.transform(wine)

    prediction_lr = modello_lr.predict(wine_scaled)[0]
    prediction_knn = modello_knn.predict(wine_scaled)[0]

    return {
        "prediction_logistic_regression": class_names[int(prediction_lr)],
        "prediction_knn": class_names[int(prediction_knn)]
    }