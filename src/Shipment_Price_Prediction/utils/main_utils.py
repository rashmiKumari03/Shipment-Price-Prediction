import os
import sys
import shutil
import dill
import yaml
import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost

from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
from xgboost import __dict__ as xgb_dict
from typing import Dict, Tuple, List, Union, Any
from sklearn.metrics import r2_score , mean_squared_error, mean_absolute_error

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE

class MainUtils:
    
    # Read YAML file and return data as dictionary
    def read_yaml_file(self, filename: str) -> dict:
            logging.info("Entered the read_yaml_file method of MainUtils class")
            try:
                # Ensure the directory exists, but don't open a non-existing file
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"The file {filename} does not exist.")
                
                # Open the YAML file in text mode
                with open(filename, "r") as yaml_file:
                    return yaml.safe_load(yaml_file)
            except Exception as e:
                raise CustomException(f"Error reading YAML file: {str(e)}", sys)
            
# Write JSON-like dictionary to a YAML file
    def write_json_to_yaml_file(self, json_file: dict, yaml_file_path: str) -> yaml:
        logging.info("Entered the write_json_to_yaml_file method of MainUtils class")
        try:
            data = json_file
            stream = open(yaml_file_path,"w")
            yaml.dump(data, stream)
        except Exception as e:
            raise CustomException(f"Error writing YAML file: {str(e)}", sys)

    # Save numpy array
    def save_numpy_array_data(self, file_path: str, array: np.array):
        logging.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logging.info("Exited the save_numpy_array_data method of MainUtils class")
            return file_path
        
        except Exception as e:
            raise CustomException(f"Error saving numpy array: {str(e)}", sys)

    # Load numpy array
    def load_numpy_array_data(self, file_path: str) -> np.array:
        logging.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj,allow_pickle=True)
        except Exception as e:
            raise CustomException(f"Error loading numpy array: {str(e)}", sys)

    # Get tuned model
    def get_tunned_model(self, model_name: str, train_x: DataFrame, train_y: DataFrame, test_x: DataFrame, test_y: DataFrame) -> Tuple[float, object, str]:
        logging.info("Entered the get_tunned_model method of MainUtils class")
        try:
            model = self.get_base_model(model_name)
            model_best_params = self.get_model_params(model, train_x, train_y)
            model.set_params(**model_best_params)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            model_all_metrics = self.get_model_score(test_y, preds)
            model_r2_score = model_all_metrics["R2 Score"]
            
            logging.info("Exited the get_tuned_model method of MainUtils class")
            
            return model_r2_score, model, model.__class__.__name__
        
        except Exception as e:
            raise CustomException(f"Error tuning model {model_name}: {str(e)}", sys)
        
        
    # Get model score
    @staticmethod
    def get_model_score(test_y : DataFrame , preds : DataFrame) -> dict:
        logging.info("Entered the get_model_score method of MainUtils class")
        try:
            # Calculate metrics
            r2 = r2_score(test_y, preds)
            mse = mean_squared_error(test_y, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_y, preds)

            metrics = {
                "R2 Score": r2,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae
            }

            logging.info(f"Model Metrics: {metrics}")
            logging.info("Exited the get_model_scores method of MainUtils class")
            
            return metrics
        except Exception as e:
            raise CustomException(f"Error in getting model score :{str(e)}",sys)
        

    # Get base model
    @staticmethod
    def get_base_model(model_name: str) -> object:
        logging.info("Entered the get_base_model method of MainUtils class")
        try:
            if model_name.lower().startswith("xgb"):
                return xgboost.__dict__.get(model_name, lambda: None)()
            all_models = dict(all_estimators())
            return all_models.get(model_name)()
        except Exception as e:
            raise CustomException(f"Error retrieving base model {model_name}: {str(e)}", sys)
        

    # Get model params using GridSearchCV
    def get_model_params(self, model: object, x_train: DataFrame, y_train: DataFrame) -> Dict:
        logging.info("Entered the get_model_params method of MainUtils class")
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

    # Save object with dill
    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(f"Error saving object: {str(e)}", sys)
        
        
    @staticmethod
    def get_best_model_with_name_and_score(model_list : list) -> Tuple[object,float]:
        logging.info("Entered the get_best_model_with_name_and_score method of MainUtils class")
        try:
            if model_list:
                logging.info("Model list is empty.")
                # Find the best model based on the score
                best_model_tuple = max(model_list, key=lambda x: x[2])  # x[2] is the score

                best_model_score = best_model_tuple[0]
                best_model_object = best_model_tuple[1]
                best_model_name = best_model_tuple[2]

                logging.info(
                    f"Best model: {best_model_name} with score: {best_model_score}"
                )
                logging.info("Exited the get_best_model_with_name_and_score method of MainUtils class")
                
                return best_model_name , best_model_score , best_model_object

        except Exception as e:
            raise CustomException(str(e),sys)
    

    @staticmethod
    def load_object(file_path:str) -> object:
        logging.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path,"rb") as file_obj:
                obj = dill.load(file_obj)
            logging.info("Exited the load_object method of MainUtils class")
            return obj
        
        except Exception as e:
            raise CustomException(str(e),sys)
        
        
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


    # Update model score in YAML
    def update_model_score(self, best_model_score: float) -> None:
        logging.info("Entered the update_model_score method of MainUtils class")
        try:
            model_config = self.read_yaml_file(MODEL_CONFIG_FILE)
            model_config["base_model_score"] = str(best_model_score)
            with open(MODEL_CONFIG_FILE, "w") as fp:
                yaml.safe_dump(model_config, fp, sort_keys=False)
        except Exception as e:
            raise CustomException(f"Error updating model score: {str(e)}", sys)
