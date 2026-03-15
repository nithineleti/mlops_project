import logging
import yaml
import mlflow
import mlflow.sklearn

from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def train_with_mlflow():

    # Load configuration
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    target_column = config["data"]["target_column"]

    import os
    os.makedirs("mlruns", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Model Training Experiment")


    with mlflow.start_run() as run:

        try:

            # -------------------------
            # DATA INGESTION
            # -------------------------
            ingestion = Ingestion()
            train, test = ingestion.load_data()

            logging.info("Data ingestion completed successfully")

            # -------------------------
            # DATA CLEANING
            # -------------------------
            cleaner = Cleaner()

            train_data = cleaner.clean_data(train)
            test_data = cleaner.clean_data(test)

            logging.info("Data cleaning completed successfully")

            # -------------------------
            # MODEL TRAINING
            # -------------------------
            trainer = Trainer()

            X_train, y_train = trainer.feature_target_separator(train_data)

            trainer.train_model(X_train, y_train)
            trainer.save_model()

            logging.info("Model training completed successfully")

            # -------------------------
            # MODEL EVALUATION
            # -------------------------
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            # ensure numeric target
            y_test = y_test.astype(int)

            y_pred = trainer.pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            logging.info("Model evaluation completed successfully")

            # -------------------------
            # MLFLOW LOGGING
            # -------------------------
            mlflow.set_tag("Model developer", "prsdm")
            mlflow.set_tag("pipeline", "sklearn pipeline")

            mlflow.log_params(config["model"]["params"])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])

            mlflow.sklearn.log_model(trainer.pipeline, "model")

            # register model
            model_name = "insurance_model"
            model_uri = f"runs:/{run.info.run_id}/model"

            mlflow.register_model(model_uri, model_name)

            logging.info("MLflow tracking completed successfully")

            # -------------------------
            # PRINT RESULTS
            # -------------------------
            print("\n============= Model Evaluation Results ==============")
            print(f"Model: {trainer.model_name}")
            print(f"Accuracy Score: {accuracy:.4f}")
            print(f"ROC AUC Score: {roc:.4f}")
            print(classification_report(y_test, y_pred))
            print("=====================================================")

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    train_with_mlflow()
