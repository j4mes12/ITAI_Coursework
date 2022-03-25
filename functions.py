import params as p
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Feature Engineering


def load_split_datasets(part: str):

    train = joblib.load(
        p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_train.pkl"
    )
    test = joblib.load(p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_test.pkl")
    val = joblib.load(p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_val.pkl")

    return train, test, val


def save_split_datasets(datasets: dict, part: str):

    train = datasets["train"]
    test = datasets["test"]
    val = datasets["val"]

    joblib.dump(
        train, p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_train.pkl"
    )
    joblib.dump(test, p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_test.pkl")
    joblib.dump(val, p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_val.pkl")


# Modelling


def split_data_X_y(data: pd.DataFrame):

    X, y = data[p.CORE_MODEL_FEATURES], data[p.MODEL_RESPONSE]

    return X, y


def evaluate_model(model, X: pd.DataFrame, y: np.array, metric: str):

    y_pred = model.predict(X)

    args = {"y_true": y, "y_pred": y_pred}

    metrics = {
        "rmse": (mean_squared_error(**args)) ** 0.5,
        "mae": [mean_absolute_error(**args)],
        "r2": [r2_score(**args)],
    }

    return metrics[metric]


def save_model(model, model_name: str):

    joblib.dump(model, p.OUTPUTS_PATH + f"model_{model_name}.pkl")


def save_preds(model, model_name: str, X: pd.DataFrame):

    y_pred = pd.DataFrame(model.predict(X))

    y_pred.to_csv(p.OUTPUTS_PATH + f"preds_{model_name}.csv")


def save_model_and_preds(model, model_name: str, X: pd.DataFrame):

    save_model(model, model_name)

    save_preds(model, model_name, X)
