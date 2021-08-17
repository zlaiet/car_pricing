import pandas as pd
import uvicorn
import pickle
import numpy as np
import warnings
import nest_asyncio

from datetime import date
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")


def normalisation(df):
    """
    This function return Encodage of each given string(carrosorie,energie,..)
    example: user take energie as 'essence' this function return encodage of 'essence'==> 1
    """
    # num_col = ["kilometrage", "puissance", "age"]
    cat_cols = [
        "carrosserie",
        "energie",
        "boite",
        "transmission",
        "couleur",
        "brand",
        "model",
    ]
    columnsValue = {}
    for col in cat_cols:
        column_labels = list(range(1, len(df[col].unique()) + 1))
        rep = {}
        for i, j in zip(column_labels, df[col].unique()):
            rep[j] = i
        columnsValue[col] = list(df[col].unique())
        df[col].replace(rep, inplace=True)

    carrosserie_ = columnsValue.get("carrosserie")  # .sort()
    energie_ = columnsValue.get("energie")  # .sort()
    boite_ = columnsValue.get("boite")  # .sort()
    transmission_ = columnsValue.get("transmission")  # .sort()
    couleur_ = columnsValue.get("couleur")  # .sort()
    brand_ = columnsValue.get("brand")  # .sort()
    model_ = columnsValue.get("model")  # .sort()
    carrosserie_.sort()
    energie_.sort()
    boite_.sort()
    transmission_.sort()
    couleur_.sort()
    brand_.sort()
    model_.sort()

    return (
        carrosserie_,
        energie_,
        boite_,
        transmission_,
        couleur_,
        brand_,
        model_,
    )


class CarPrice(BaseModel):
    """ here we define all params with type, in this clas we try to check if given value type correct or not"""

    kilometrage: float
    carrosserie: str
    energie: str
    puissance: int
    boite: str
    transmission: str
    couleur: str
    model: str
    age: int
    brand: str


# nest_asyncio.apply()

# load data
data = pd.read_csv(
    "data/data_after_preprocessing.csv",
    names=[
        "kilometrage",
        "carrosserie",
        "energie",
        "puissance",
        "boite",
        "transmission",
        "couleur",
        "model",
        "age",
        "brand",
        "price",
    ],
    encoding="utf-8",
    error_bad_lines=False,
)

# load scaller model
standardscaler = pickle.load(open("checkpoint/StandardScaler.sav", "rb"))


# OneHotencode of params values
(
    carrosserie_,
    energie_,
    boite_,
    transmission_,
    couleur_,
    brand_,
    model_,
) = normalisation(data)

# Begin FastAPI
app = FastAPI()


@app.on_event("startup")
def load_model():
    """ load saved model"""
    global predict_model
    predict_model = pickle.load(open("checkpoint/AdaBoostRegressor.pkl", "rb"))


@app.get("/")
def index():
    return {"message": "This is the homepage of the API "}


@app.post("/prediction")
def predict(received_data: CarPrice):
    """ return price of given params """
    received = received_data.dict()

    # get age
    year = received["age"]
    todays_date = date.today()
    current_year = todays_date.year
    # claculate year
    age = current_year - year
    kilometrage = received["kilometrage"]
    puissance = received["puissance"]

    # scale integer value
    km_p_age = pd.DataFrame(
        data=[[kilometrage, puissance, age]],
        columns=["kilometrage", "puissance", "age"],
    )
    x = standardscaler.transform(
        km_p_age[["kilometrage", "puissance", "age"]]
    ).flatten()

    # get user value
    carrosserie = received["carrosserie"]
    energie = received["energie"]
    boite = received["boite"]
    transmission = received["transmission"]
    couleur = received["couleur"]
    brand = received["brand"]
    model = received["model"]+" "

    enc_carrosserie = carrosserie_.index(carrosserie)
    enc_energie = energie_.index(energie)
    enc_boite = boite_.index(boite)
    enc_transmission = transmission_.index(transmission)
    enc_couleur = couleur_.index(couleur)
    enc_brand = brand_.index(brand)
    enc_model = model_.index(model)

    # get columns
    X_columns = data.iloc[:, :-1]

    # prepare dataframe to make prediction
    test = pd.DataFrame(
        data=[
            [
                x[0],
                enc_carrosserie,
                enc_energie,
                x[1],
                enc_boite,
                enc_transmission,
                enc_couleur,
                enc_model,
                x[2],
                enc_brand,
            ]
        ],
        columns=X_columns.columns,
    )
    # predict and retur results
    pred_name = predict_model.predict(test)
    price = np.exp(pred_name[0])
    return {"prediction": price}


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=2200, debug=True)
