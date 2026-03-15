import pandas as pd
import joblib
import yaml


def evaluate_model():

    # load config
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    target_column = config["data"]["target_column"]
    model_path = config["model"]["store_path"] + "/model.pkl"

    # load model pipeline
    model = joblib.load(model_path)

    # load test data
    df = pd.read_csv("data/clean_test.csv")

    # split features and target
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # predictions
    y_pred = model.predict(X_test)

    print("Predictions completed")

    return y_test, y_pred
