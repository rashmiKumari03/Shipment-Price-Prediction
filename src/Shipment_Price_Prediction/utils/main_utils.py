import os
import sys
import shutil
import joblib
import yaml
import numpy as np
from pandas import DataFrame
from typing import Dict, Tuple, Optional
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE


class MainUtils:
    # ============================================================
    # ======== CONFIGURATION FILE HANDLING (YAML) ===============
    # ============================================================

    # I use this to read YAML files (like my config file)
    def read_yaml_file(self, filename: str) -> dict:
        logging.info("Entered read_yaml_file")
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"The file {filename} does not exist.")
            with open(filename, "r") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            raise CustomException(f"Error reading YAML file: {str(e)}", sys)

    # I use this to write my dictionary data back into a YAML file
    def write_json_to_yaml_file(self, json_file: dict, yaml_file_path: str) -> None:
        logging.info("Entered write_json_to_yaml_file")
        try:
            with open(yaml_file_path, "w") as stream:
                yaml.dump(json_file, stream)
        except Exception as e:
            raise CustomException(f"Error writing YAML file: {str(e)}", sys)

    # I use this to update the model score in the config file
    def update_model_score(self, best_model_score: float) -> None:
        logging.info("Entered update_model_score")
        try:
            model_config = self.read_yaml_file(MODEL_CONFIG_FILE)
            model_config["base_model_score"] = str(best_model_score)
            with open(MODEL_CONFIG_FILE, "w") as fp:
                yaml.safe_dump(model_config, fp, sort_keys=False)
        except Exception as e:
            raise CustomException(f"Error updating model score: {str(e)}", sys)

    # ============================================================
    # ======== NUMPY DATA HANDLING ===============================
    # ============================================================

    # I use this to save a numpy array to disk
    def save_numpy_array_data(self, file_path: str, array: np.ndarray) -> str:
        logging.info("Entered save_numpy_array_data")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            return file_path
        except Exception as e:
            raise CustomException(f"Error saving numpy array: {str(e)}", sys)

    # I use this to load a numpy array from disk
    def load_numpy_array_data(self, file_path: str) -> np.ndarray:
        logging.info("Entered load_numpy_array_data")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj, allow_pickle=True)
        except Exception as e:
            raise CustomException(f"Error loading numpy array: {str(e)}", sys)

    # ============================================================
    # ======== MODEL BUILDING AND TUNING =========================
    # ============================================================

    # I use this to get a base model by name (either xgboost or sklearn)
    @staticmethod
    def get_base_model(model_name: str) -> object:
        logging.info("Entered get_base_model")
        try:
            # For sklearn models
            all_models = dict(all_estimators())
            return all_models.get(model_name)()
        except Exception as e:
            raise CustomException(f"Error retrieving base model {model_name}: {str(e)}", sys)

    # I use this to perform GridSearchCV and get the best hyperparameters
    def get_model_params(self, model: object, x_train: DataFrame, y_train: DataFrame) -> Dict:
        logging.info("Entered get_model_params")
        try:
            model_name = model.__class__.__name__
            model_config = self.read_yaml_file(MODEL_CONFIG_FILE)
            model_param_grid = model_config.get("train_model", {}).get(model_name)
            if not model_param_grid:
                raise ValueError(f"No hyperparameters found for {model_name} in config file.")
            model_grid = GridSearchCV(model, model_param_grid, cv=2, verbose=3, n_jobs=-1)
            model_grid.fit(x_train, y_train)
            return model_grid.best_params_
        except Exception as e:
            raise CustomException(f"Error during parameter tuning: {str(e)}", sys)

    # I use this to train with best params, do cross-validation, and evaluate
    def get_tunned_model(
        self,
        model_name: str,
        train_x: DataFrame,
        train_y: DataFrame,
        test_x: DataFrame,
        test_y: DataFrame
    ) -> Tuple[float, object, str, Dict]:
        logging.info("Entered get_tunned_model")
        try:
            model = self.get_base_model(model_name)
            model_best_params = self.get_model_params(model, train_x, train_y)
            model.set_params(**model_best_params)

            cv_scores = cross_val_score(model, train_x, train_y, cv=5)
            logging.info(f"Cross-validation scores: {cv_scores}")
            logging.info(f"Mean CV score: {np.mean(cv_scores)}")

            model.fit(train_x, train_y)
            preds = model.predict(test_x)

            model_all_metrics = self.get_model_score(test_y, preds)
            model_r2_score = model_all_metrics["R2 Score"]

            return model_r2_score, model, model.__class__.__name__, model_all_metrics
        except Exception as e:
            raise CustomException(f"Error tuning model {model_name}: {str(e)}", sys)

    # I use this to calculate all metrics for my model
    @staticmethod
    def get_model_score(test_y: DataFrame, preds: DataFrame) -> Dict:
        logging.info("Entered get_model_score")
        try:
            r2 = np.round(r2_score(test_y, preds), 4)
            mse = np.round(mean_squared_error(test_y, preds), 4)
            rmse = np.round(np.sqrt(mse), 4)
            mae = np.round(mean_absolute_error(test_y, preds), 4)

            metrics = {"R2 Score": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}
            logging.info(f"Model Metrics: {metrics}")
            return metrics
        except Exception as e:
            raise CustomException(f"Error in getting model score: {str(e)}", sys)

    # I use this to choose the best model out of a list of models with scores
    @staticmethod
    def get_best_model_with_name_and_score(model_list: list) -> Tuple[str, object, float, dict]:
        logging.info("Entered get_best_model_with_name_and_score")
        try:
            if model_list:
                best_model_tuple = max(model_list, key=lambda x: x[0])
                best_model_score = best_model_tuple[0]
                best_model_object = best_model_tuple[1]
                best_model_name = best_model_tuple[2]
                best_model_metrics = best_model_tuple[3]
                logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
                return best_model_name, best_model_object, best_model_score, best_model_metrics
        except Exception as e:
            raise CustomException(str(e), sys)

    # ============================================================
    # ======== OBJECT SERIALIZATION (SAVE/LOAD) ==================
    # ============================================================

    # I use this to save any object (like a model) to a file
    @staticmethod
    def save_object(file_path: str, obj: object) -> str:
        logging.info("Entered save_object")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(obj, file_path)
            return file_path
        except Exception as e:
            raise CustomException(f"Error saving object: {str(e)}", sys)

    # I use this to load any object (like a model) from a file
    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered load_object")
        try:
            return joblib.load(file_path)
        except Exception as e:
            raise CustomException(f"Error loading object: {str(e)}", sys)

    # ============================================================
    # ======== FILE AND FOLDER UTILITIES =========================
    # ============================================================

    # I use this to zip a folder (create a zip archive)
    @staticmethod
    def create_artifacts_zip(file_name: str, folder_name: str) -> None:
        logging.info("Entered create_artifacts_zip")
        try:
            shutil.make_archive(file_name, "zip", folder_name)
        except Exception as e:
            raise CustomException(str(e), sys)

    # I use this to unzip a zip file into a folder
    @staticmethod
    def unzip_file(filename: str, folder_name: str) -> None:
        logging.info("Entered unzip_file")
        try:
            shutil.unpack_archive(filename, folder_name)
        except Exception as e:
            raise CustomException(str(e), sys)

    # ============================================================
    # ======== SAFE TYPE CONVERSIONS =============================
    # ============================================================

    # I use this to safely convert string to int
    @staticmethod
    def safe_int(value: str) -> Optional[int]:
        try:
            return int(value) if value.strip() else None
        except ValueError:
            return None

    # I use this to safely convert string to float
    @staticmethod
    def safe_float(value: str) -> Optional[float]:
        try:
            return float(value) if value.strip() else None
        except ValueError:
            return None

    # I use this to clean a string so it can be used as a key (for MLflow)
    @staticmethod
    def sanitize_key(s: str) -> str:
        return s.translate(str.maketrans({
            " ": "_", "/": "_", "\\": "_",
            ":": "_", ".": "_", "%": "_",
            "\"": "_", "'": "_"
        }))
