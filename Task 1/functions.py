import params as p
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Feature Engineering


def load_split_datasets(part: str):
    """Loads the train, test and validation datasets from the data folder

    Args:
        part (str): code section we want the saved data from

    Returns:
        tuple: returns the loaded datasets
    """

    train = joblib.load(
        p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_train.pkl"
    )
    test = joblib.load(p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_test.pkl")
    val = joblib.load(p.TTV_DATA_PATH + f"{part}_{p.DATA_SAVE_NAME}_val.pkl")

    return (train, test, val)


def save_split_datasets(datasets: dict, part: str):
    """Saves the train, test and validation datasets to the data folder

    Args:
        datasets (dict): dictionary of datasets with the structure
            {'train' : train_set, 'test' : test_set, 'val' : val_set}
        part (str): code section we want to associate the saved data to
    """

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
    """Splits the data into X and y using the defined features

    Args:
        data (pd.DataFrame): Data that we want to split

    Returns:
        tuple: X and y split dataframes
    """

    X, y = data[p.CORE_MODEL_FEATURES], data[p.MODEL_RESPONSE]

    return (X, y)


def evaluate_model(model, X: pd.DataFrame, y: np.array, metric: str):
    """Calculates the rmse, mae and r2 scores for a specified model,
    rounding the results to six decimal places

    Args:
        model: model we want to evaluate
        X (pd.DataFrame): dataframe we want to use to make predictions
        y (np.array): array we use to assess our predictions
        metric (str): define which metric we want to output

    Returns:
        float: specified metric
    """

    y_pred = model.predict(X)

    args = {"y_true": y, "y_pred": y_pred}

    metrics = {
        "rmse": (mean_squared_error(**args)) ** 0.5,
        "mae": mean_absolute_error(**args),
        "r2": r2_score(**args),
    }

    metrics = {k: round(v, 6) for k, v in metrics.items()}

    return metrics[metric]


def save_model(model, model_name: str):
    """Saved specified model to outputs folder

    Args:
        model : model to be saved
        model_name (str): name of model
    """

    joblib.dump(model, p.OUTPUTS_PATH + f"model_{model_name}.pkl")


def save_preds(model, model_name: str, X: pd.DataFrame):
    """Calculates and saves predictions made by the model

    Args:
        model: model to predict
        model_name (str): name of model
        X (pd.DataFrame): dataframe used to make predictions
    """

    y_pred = pd.DataFrame(model.predict(X))

    y_pred.to_csv(p.OUTPUTS_PATH + f"preds_{model_name}.csv")


def save_model_and_preds(model, model_name: str, X: pd.DataFrame):
    """Calculates model predictions and saves both model and predictions
    to output folder. This uses the save_preds and save_model functions.

    Args:
        model (_type_): model to predict and save
        model_name (str): name of model
        X (pd.DataFrame): dataframe used to make predictions
    """

    save_model(model, model_name)

    save_preds(model, model_name, X)


def load_models():
    """Loads all the .pkl files that that have the model suffix in
    the outputs folder

    Returns:
        output_models (tuple): tuple of loaded models
    """

    models = {}

    for file in os.listdir(p.OUTPUTS_PATH):

        file_split = file.split("_")

        if file_split[-1] == "model.pkl":
            models[file_split[0]] = joblib.load(p.OUTPUTS_PATH + file)

    output_models = (models["baseline"], models["xgb"], models["rf"])

    return output_models
