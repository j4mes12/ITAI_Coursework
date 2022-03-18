import params as p
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error

# Modelling


def split_data_X_y(data: pd.DataFrame):

    X, y = data[p.CORE_MODEL_FEATURES], data[p.MODEL_RESPONSE]

    return X, y


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.array):

    y_pred = model.predict(X_test)

    print(f"RMSE: {(mean_squared_error(y_true=y_test, y_pred=y_pred))**0.5}")


def save_model(model, model_name: str):

    joblib.dump(model, p.OUTPUTS_PATH + f"model_{model_name}.pkl")


def save_preds(model, model_name: str, X_test):

    y_pred = pd.DataFrame(model.predict(X_test))

    y_pred.to_csv(p.OUTPUTS_PATH + f"preds_{model_name}.csv")


def save_model_and_preds(model, model_name: str, X_test):

    save_model(model, model_name)

    save_preds(model, model_name, X_test)
