import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
import pickle

# libraries for preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# libraries for evaluation
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# libraries for models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


warnings.filterwarnings("ignore")


def add(price):
    return price + 1000


def reduce(price):
    return price - 1000


def add_km(km):
    return km + 3000


def reduce_km(km):
    return km - 3000


def data_augmentation(df):
    """ we try here to augment data to get perfermance model but we not sure """
    df_2 = df.copy()
    df_3 = df.copy()
    df_2["kilometrage"] = df_2.price.apply(add_km)
    df_2["price"] = df_2.price.apply(add)

    df_3["kilometrage"] = df_3.price.apply(reduce_km)
    df_3["price"] = df_3.price.apply(reduce)

    df = pd.concat([df, df_2, df_3])
    return df


def scaller_model(data_2):
    """ function to scale num column [-1,1] """
    # Scaller model
    norm = StandardScaler()
    norm.fit(data_2[["kilometrage", "puissance", "age"]])
    # standardvalues = norm.transform(data_2[["kilometrage", "puissance", "age"]])
    print("save scaller model")
    pickle.dump(norm, open("checkpoint/StandardScaler.sav", "wb"))


def trainingData(df, n):
    """ function to split dataset into training and test """
    X = df.iloc[:, n]
    y = df.iloc[:, -1:].values.T
    y = y[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, test_size=0.1, random_state=0
    )
    return (X_train, X_test, y_train, y_test)


def remove_neg(y_test, y_pred):
    """ some of models will predict neg values so this function will remove that values """
    ind = [index for index in range(len(y_pred)) if (y_pred[index] > 0)]
    y_pred = y_pred[ind]
    y_test = y_test[ind]
    y_pred[y_pred < 0]
    return (y_test, y_pred)


def result(y_test, y_pred):
    """ Evaluate the given model results """
    r = []
    r.append(mean_squared_log_error(y_test, y_pred))
    r.append(np.sqrt(r[0]))
    r.append(r2_score(y_test, y_pred))
    r.append(round(r2_score(y_test, y_pred) * 100, 4))
    return r


def train(data, metrics_dir):
    """ train four model and save who have the best accuracy """

    # define numerical and categorical columns
    # num_col = ["kilometrage", "puissance", "price", "age"]
    cat_cols = [
        "carrosserie",
        "energie",
        "boite",
        "transmission",
        "couleur",
        "brand",
        "model",
    ]

    # label encoder, example boite automatique code is 0
    le = preprocessing.LabelEncoder()
    data[cat_cols] = data[cat_cols].apply(le.fit_transform)

    # scaling numerical data
    norm = StandardScaler()
    data["price"] = np.log(data["price"])
    data["kilometrage"] = norm.fit_transform(
        np.array(data["kilometrage"]).reshape(-1, 1)
    )
    data["puissance"] = norm.fit_transform(np.array(data["puissance"]).reshape(-1, 1))
    data["age"] = norm.fit_transform(np.array(data["age"]).reshape(-1, 1))
    # scaling target variable
    q1, q3 = data["price"].quantile([0.25, 0.75])
    o1 = q1 - 1.5 * (q3 - q1)
    o2 = q3 + 1.5 * (q3 - q1)
    data = data[(data.price >= o1) & (data.price <= o2)]

    # split data
    X_train, X_test, y_train, y_test = trainingData(
        data, list(range(len(list(data.columns)) - 1))
    )

    # dataframe that store the performance of each model
    accu = pd.DataFrame(index=["MSLE", "Root MSLE", "R2 Score", "Accuracy(%)"])

    # 1- LinearRegression model
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    # calculating error/accuracy
    y_test_1, y_pred_1 = remove_neg(y_test, y_pred)
    r1_lr = result(y_test_1, y_pred_1)
    # save metrics
    accu["Linear Regression"] = r1_lr
    accu.to_csv(metrics_dir)

    # 2- AdaBoostRegressor model
    ABR = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=24), n_estimators=200, learning_rate=0.6
    )
    ABR.fit(X_train, y_train)
    y_pred = ABR.predict(X_test)
    # model evaluation
    r7_ab = result(y_test, y_pred)
    # save metrics
    accu["AdaBoost Regressor"] = r7_ab
    accu.to_csv(metrics_dir)

    # 3- XGBoost Regressor model
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.2,
        max_depth=24,
        alpha=5,
        n_estimators=200,
    )
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)
    # model evaluation
    y_test_1, y_pred_1 = remove_neg(y_test, y_pred)
    r8_xg = result(y_test_1, y_pred_1)
    # save metrics
    accu["XGBoost Regressor"] = r8_xg
    accu.to_csv(metrics_dir)

    # 4- RandomForestRegressor model
    RFR = RandomForestRegressor(
        n_estimators=150,
        random_state=0,
        min_samples_leaf=1,
        max_features=0.5,
        n_jobs=-1,
        oob_score=True,
    )
    RFR.fit(X_train, y_train)
    y_pred = RFR.predict(X_test)
    r5_rf = result(y_test, y_pred)
    # save metrics
    accu["RandomForest Regressor"] = r5_rf
    accu.to_csv(metrics_dir)

    # save Results
    accu = pd.read_csv(metrics_dir, index_col=0)
    accu

    model_accuracy = accu.loc["Accuracy(%)"]
    x = list(range(len(model_accuracy)))
    y = list(range(0, 101, 10))
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.figure(figsize=(20, 6))
    plt.plot(model_accuracy)
    plt.yticks(y)
    plt.xticks(fontsize=20)
    plt.xticks(rotation=(10))
    plt.xlabel("Models", fontsize=30)
    plt.ylabel("Accuracy(%)", fontsize=30)
    plt.title("Performance of Models")
    for a, b in zip(x, y):
        b = model_accuracy[a]
        val = "(" + str(round(model_accuracy[a], 2)) + " %)"
        plt.text(
            a,
            b + 4.5,
            val,
            horizontalalignment="center",
            verticalalignment="center",
            color="green",
            bbox=props,
        )
        plt.text(
            a,
            b + 3.5,
            ".",
            horizontalalignment="center",
            verticalalignment="center",
            color="red",
            fontsize=50,
        )
    plt.tight_layout()
    plt.savefig("metrics/Overall-Performance.jpg", dpi=600)
    # plt.show()

    # save model
    # open a file, where you ant to store the data
    file = open("checkpoint/AdaBoostRegressor.pkl", "wb")
    pickle.dump(ABR, file)


def main():
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
        encoding="latin1",
        error_bad_lines=False,
    )
    metrics_dir = "metrics/error.csv"
    # 1- augmenter dataset
    data = data_augmentation(data)
    # 2- run scaller model and save it
    scaller_model(data)
    # 3- train models and save results
    print("start training for four models ...")
    train(data, metrics_dir)
    print("training completed :) ")
    print(
        "====================================== Results ==================================="
    )
    accu = pd.read_csv(metrics_dir, index_col=0)
    print(accu)


if __name__ == "__main__":
    main()
