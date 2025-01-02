import shutil
import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

import xgboost
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators
from yaml import safe_dump

from typing import Dict, Tuple, List
import dill
import yaml

from src.Shipment_Price_Prediction.constant import *


class MainUtils:
    # Read YAML file and return data as dictionary
    def read_yaml_file(self, filename: str) -> dict:
        logging.info("Entered the read_yaml_file method of MainUtils class")
        
        try:
            # Open and load the YAML file into a dictionary
            with open(filename, "r") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Convert a dictionary (JSON-like structure) to YAML and write to file
    def write_json_to_yaml_file(self, json_file: dict, yaml_file_path: str) -> None:
        logging.info("Entered the write_json_to_yaml_file method of MainUtils class")
        
        try:
            # Open the YAML file for writing and dump the JSON content as YAML
            with open(yaml_file_path, "w") as stream:
                yaml.dump(json_file, stream)
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Save a numpy array to a specified file path in binary format
    def save_numpy_array_data(self, file_path: str, array: np.array) -> str:
        logging.info("Entered the save_numpy_array_data method of MainUtils class")
        
        try:
            # Write numpy array to a binary file
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logging.info("Exited the save_numpy_array_data method of MainUtils class")
            return file_path
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Load numpy array data from a binary file
    def load_numpy_array_data(self, file_path: str) -> np.array:
        logging.info("Entered the load_numpy_array_data method of MainUtils class")
        
        try:
            # Read numpy array from binary file
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Get the tuned machine learning model based on the given model name and training data
    def get_tunned_model(self, model_name: str, train_x: DataFrame, train_y: DataFrame,
                         test_x: DataFrame, test_y: DataFrame) -> Tuple[float, object, str]:
        logging.info("Entered the get_tunned_model method of MainUtils class")
        
        try:
            # Get the base model
            model = self.get_base_model(model_name)
            
            # Get best hyperparameters for the model
            model_best_params = self.get_model_params(model, train_x, train_y)
            
            # Set the best parameters and fit the model
            model.set_params(**model_best_params)
            model.fit(train_x, train_y)
            
            # Make predictions on the test set
            preds = model.predict(test_x)
            
            # Calculate and return model score (R-squared)
            model_score = self.get_model_score(test_y, preds)
            logging.info("Exited the get_tunned_model method of MainUtils class")
            
            return model_score, model, model.__class__.__name__
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Calculate the R-squared score of the model predictions
    @staticmethod
    def get_model_score(test_y: DataFrame, preds: DataFrame) -> float:
        logging.info("Entered the get_model_score method of MainUtils class")
        
        try:
            # Compute R-squared score for model evaluation
            model_score = r2_score(test_y, preds)
            logging.info(f"Model Score is {model_score}")
            logging.info("Exited the get_model_score method of MainUtils class")
            
            return model_score
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Get the base model object from the provided model name
    @staticmethod
    def get_base_model(model_name: str) -> object:
        logging.info("Entered the get_base_model method of MainUtils class")
        
        try:
            # Load model using model_name, supporting both xgboost and sklearn models
            if model_name.lower().startswith("xgb"):
                model = xgboost.__dict__[model_name]()
            else:
                model_idx = [model[0] for model in all_estimators()].index(model_name)
                model = all_estimators().__getitem__(model_idx)[1]()
            logging.info("Exited the get_base_model method of MainUtils class")
            return model
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Get the best hyperparameters for the model using GridSearchCV
    @staticmethod
    def get_model_params(self, model: object, x_train: DataFrame, y_train: DataFrame) -> Dict:
        logging.info("Entered the get_model_params method of MainUtils class")
        
        try:
            # Set up GridSearchCV with cross-validation and verbose settings
            VERBOSE = 3
            CV = 2
            N_JOBS = -1
            
            # Read the model configuration file to get hyperparameters
            model_name = model.__class__.__name__
            model_config = self.read_yaml_file(filename=MODEL_CONFIG_FILE)
            model_param_grid = model_config["train_model"][model_name]
            
            # Perform grid search to find best parameters
            model_grid = GridSearchCV(model, model_param_grid, verbose=VERBOSE, cv=CV, n_jobs=N_JOBS)
            model_grid.fit(x_train, y_train)
            logging.info("Exited the get_model_params method of MainUtils class")
            
            return model_grid.best_params_
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Save a Python object to a specified file using dill serialization
    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")
        
        try:
            # Serialize and save the object to file
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)
            logging.info("Exited the save_object method of MainUtils class")
            return file_path
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Get the best model from a list of models based on score
    @staticmethod
    def get_best_model_with_name_and_score(model_list: list) -> Tuple[object, float]:
        logging.info("Entered the get_best_model_with_name_and_score method of MainUtils class")
        
        try:
            # Find the model with the highest score in the list
            best_score = max(model_list)[0]
            best_model = max(model_list)[1]
            logging.info("Exited the get_best_model_with_name_and_score method of MainUtils class")
            
            return best_model, best_score
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Load a Python object from a specified file using dill deserialization
    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")
        
        try:
            # Deserialize and load the object from file
            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)
            logging.info("Exited the load_object method of MainUtils class")
            return obj
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Create a zip archive of a specified folder
    @staticmethod
    def create_artifacts_zip(file_name: str, folder_name: str) -> None:
        logging.info("Entered the create_artifacts_zip method of MainUtils class")
        
        try:
            # Create a zip archive of the folder
            shutil.make_archive(file_name, "zip", folder_name)
            logging.info("Exited the create_artifacts_zip method of MainUtils class")
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Unzip a specified file into a target folder
    @staticmethod
    def unzip_file(filename: str, folder_name: str) -> None:
        logging.info("Entered the unzip_file method of MainUtils class")
        
        try:
            # Unzip the archive into the specified folder
            shutil.unpack_archive(filename, folder_name)
            logging.info("Exited the unzip_file method of MainUtils class")
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)

    # Update the model score in the configuration file
    def update_model_score(self, best_model_score: float) -> None:
        logging.info("Entered the update_model_score method of MainUtils class")
        
        try:
            # Read the model config, update the base model score, and save it back
            model_config = self.read_yaml_file(filename=MODEL_CONFIG_FILE)
            model_config["base_model_score"] = str(best_model_score)
            with open(MODEL_CONFIG_FILE, "w+") as fp:
                safe_dump(model_config, fp, sort_keys=False)
            logging.info("Exited the update_model_score method of MainUtils class")
        except Exception as e:
            logging.info(CustomException(str(e), sys))  # Log error and raise custom exception
            raise CustomException(str(e), sys)
