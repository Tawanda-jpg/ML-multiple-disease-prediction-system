import pandas as pd
import numpy as np
from joblib import dump
import yaml
import logging
from pathlib import Path
from src.training.config.settings import Settings

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score
)

def train_model():
    try:
        #load .env file content to env vars
        settings = Settings()

        DATASET_PATH = Path(settings.heart_disease_model_path)
        MODEL_PATH = Path(settings.diabetes_model_path)
        LOG_PATH = Path(settings.log_path)
        HYPER_PARAMS_YAML_PATH = Path(settings.hyper_params_yaml_path)

        HEART_DISEASE_TARGET_COLUMN = settings.heart_disease_target_column
        RANDOM_STATE = settings.random_state
        TEST_SIZE = settings.test_size

        MODEL_PATH.parent.mkdir(parents = True, exist_ok = True)
        LOG_PATH.parent.mkdir(parents = True, exist_ok = True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s ",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        #LOAD DATA
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        #separate x and y
        X = df.drop(columns=[HEART_DISEASE_TARGET_COLUMN])
        y = df[HEART_DISEASE_TARGET_COLUMN]

        #crete a signature for each row to prevent duplicate leakage to test set
        row_signature = pd.util.hash_pandas_object(X, index = False)

        #group-based split
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups=row_signature))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"Train shape: {X_train.shape} and Test shape: {X_test.shape}")

        with open(HYPER_PARAMS_YAML_PATH, "r") as file:
            hyperparams = yaml.safe_load(file)

        model_params = hyperparams["heart_disease"]["params"]

        #best parameters
        best_rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **model_params
        )


        #keeping scaler in pipeline
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", best_rf)
            ]
        )

        pipeline.fit(X_train, y_train)
        logging.info("Model Training Completed")

        #model evaluation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        #accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        #recall_score
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        #f1-score
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f" Train Accuracy :{train_acc:.4f} | Train Recall Score:{train_recall:.4f} | Train f1 Score :{train_f1:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.4f} | Test Recall Score: {test_recall:.4f} | Test f1 Score:{test_f1:.4f}")

        logging.info("Train Classification Report:\n" + classification_report(y_train, y_train_pred))
        logging.info("Test Classification Report:\n" + classification_report(y_test, y_test_pred))

        #save the trained model
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to: {MODEL_PATH}")

        logging.info("Training Script Completed")

    except Exception as e:
        print(f"Training failed: {e}")
        logging.exception(f"Training Script Failed: {e}")
        raise

if __name__ == "__main__":
    train_model()
