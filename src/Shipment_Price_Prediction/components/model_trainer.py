import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Tuple
from from_root.root import from_root


from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE, SCHEMA_FILE_PATH 
from src.Shipment_Price_Prediction.components.cost_model import Cost_Model
from src.Shipment_Price_Prediction.entity.config_entity import Data_Transformation_Config , Model_Trainer_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import Data_Transformation_Artifacts, Model_Trainer_Artifacts

# Imported because the saved preprocessor object (via joblib) internally references 
# the Data_Transformation class. Even though we don’t call it directly here,
# joblib.load() needs this class to be importable in the current context.

from src.Shipment_Price_Prediction.components.data_transformation import Data_Transformation

# Imported because the saved preprocessor object also references a custom transformer
# function/class called datetime_transform_wrapper inside its pipeline. 
# When loading the object, Python must be able to locate this symbol.

from src.Shipment_Price_Prediction.components.data_transformation import datetime_transform_wrapper



import mlflow
import mlflow.sklearn
import json
from dotenv import load_dotenv
import dagshub

import warnings
warnings.filterwarnings("ignore")



class Model_Trainer:

    def __init__(self, 
                 data_transformation_artifact: Data_Transformation_Artifacts, 
                 model_trainer_config: Model_Trainer_Config):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        
         
        UTILS = MainUtils()
        self.SCHEMA_CONFIG = UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
        self.numeric_columns = self.SCHEMA_CONFIG.get('numeric_features',[])
        
        
    
    # This method is used to get the trained model.
    def get_trained_models(self, train_data: DataFrame, test_data: DataFrame) -> List[Tuple[float, object, str]]:
        """
        
        Method Name : get_trained_models
        
        Description : Retrieve trained models from config and train them.
        
        Output      : List of trained models
        """
        logging.info("Entered get_trained_models method of Model_Trainer class.")
        try:
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            models_list = list(model_config.get("train_model", {}).keys())
            logging.info(f"Got the models list from config file to train: {models_list}")
            
            '''
            Usually, splitting occurs when both the independent and dependent data are in the same training set. 
            However, in this case, since my independent and dependent data are already saved in separate files in .npz format, 
            I will load them directly."
            '''
            
            logging.info(f"Training data: {train_data.shape}, Test data: {test_data.shape}")
            
            X_train , y_train ,   X_test , y_test  = (train_data.drop(train_data.columns[len(train_data.columns)-1],axis=1) ,
                                                      train_data.iloc[:,-1] ,
                                                      test_data.drop(test_data.columns[len(test_data.columns)-1],axis=1) , 
                                                      test_data.iloc[:,-1])         
            
          
            
            trained_models = [
                self.model_trainer_config.UTILS.get_tunned_model(model_name,X_train, y_train,X_test, y_test)
                for model_name in models_list
            ]

            logging.info("Model training completed.")
            logging.info("Got Trained Model List")
            logging.info("Exited the get_trained_models method of Model_Trainer class")
            
            return trained_models

        except Exception as e:
            logging.error(f"Error in get_trained_models: {str(e)}")
            raise CustomException(str(e), sys)


    # This method is used to initialize model training
    def initiate_model_trainer(self) -> Model_Trainer_Artifacts:
        """
        Method Name : initiate_model_trainer
        
        Description : Initiate model training, select the best model, and save it.
        
        Output      : List of trained models
        """
        logging.info("Entered initiate_model_trainer method of Model_Trainer class.")
        try:
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            
            train_arr = pd.DataFrame(self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.train_file_path))
            test_arr = pd.DataFrame(self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.test_file_path))
            
            logging.info(f"X_train looks like : {train_arr.head()}")
            logging.info(f"X_test looks like : {test_arr.head()}")
            logging.info("===========================================")

            trained_models = self.get_trained_models(train_arr, test_arr)
            
            logging.info(f"trained_models:{trained_models}")
            logging.info("+++++++++++++++++++++++++++++++++++++++++++++++")
            
            best_model_name ,best_model_object, best_model_score , best_model_metrics = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(trained_models)
            
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            base_model_score = float(model_config["base_model_score"])
            
                                    
            if best_model_score >= base_model_score:
                
                self.model_trainer_config.UTILS.update_model_score(best_model_score)

                preprocessor_obj = self.model_trainer_config.UTILS.load_object(self.data_transformation_artifact.transformed_object_file_path)
                logging.info("Saving the Cost_Model using save_object")

                cost_model = Cost_Model(preprocessor_obj, best_model_object)
                logging.info(f"Created a Cost_Model: {cost_model} of type {type(cost_model)}")
            

                model_file_path = self.model_trainer_config.UTILS.save_object(self.model_trainer_config.TRAINED_MODEL_FILE_PATH, cost_model)
                logging.info(f"Model saved at {model_file_path}")
                
        
                # -------------------------- Start MLflow Logging --------------------------
                with mlflow.start_run(run_name="Model_Trainer_Run"):

                    # Tags for tracking
                    mlflow.set_tag("stage", "Model_Trainer")
                    mlflow.set_tag("pipeline", "Shipment_Price_Prediction")
                    mlflow.set_tag("best_model", best_model_name)

                    # --------------------- Intermediate Models ---------------------
                    for score, model_obj, model_name, model_metrics in trained_models:
                        if model_name == best_model_name:
                            continue  # Skip best model here

                        clean_model_name = self.model_trainer_config.UTILS.sanitize_key(model_name)
                        model_folder = f"Intermediate_Models_{clean_model_name}"

        
                        # Save metrics to JSON file
                        metrics_file = os.path.join(
                            self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,
                            f"{clean_model_name}_metrics.json"
                        )
                        with open(metrics_file, "w") as f:
                            json.dump(model_metrics, f, indent=4)
                        mlflow.log_artifact(metrics_file, artifact_path=model_folder)
                        os.remove(metrics_file)

                        # Save parameters to TXT file
                        if hasattr(model_obj, "get_params"):
                            params = model_obj.get_params()
                            params_file = os.path.join(
                                self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,
                                f"{clean_model_name}_params.txt"
                            )
                            with open(params_file, "w") as f:
                                for k, v in params.items():
                                    f.write(f"{k}: {v}\n")
                            mlflow.log_artifact(params_file, artifact_path=model_folder)
                            os.remove(params_file)

                    # --------------------- Best Model ---------------------
                    best_folder = "Best_Model_Artifact"

                    # Log model
                    mlflow.sklearn.log_model(best_model_object, artifact_path=best_folder)

                    # Save and log metrics
                    best_metrics_file = os.path.join(
                        self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,
                        "best_model_metrics.json"
                    )
                    with open(best_metrics_file, "w") as f:
                        json.dump(best_model_metrics, f, indent=4)
                    mlflow.log_artifact(best_metrics_file, artifact_path=best_folder)
                    os.remove(best_metrics_file)

                    # This is  Optional : log metrics globally for best model
                    for metric_key, metric_val in best_model_metrics.items():
                        mlflow.log_metric(f"Best_model_{self.model_trainer_config.UTILS.sanitize_key(metric_key)}", float(metric_val))

                    # Saving and loggging best model parameters
                    if hasattr(best_model_object, "get_params"):
                        best_params = best_model_object.get_params()
                        best_params_file = os.path.join(
                            self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,
                            "best_model_params.txt"
                        )
                        with open(best_params_file, "w") as f:
                            for k, v in best_params.items():
                                f.write(f"{k}: {v}\n")
                        mlflow.log_artifact(best_params_file, artifact_path=best_folder)
                        os.remove(best_params_file)
                            
                        for k, v in best_params.items():
                            if v is not None:
                                mlflow.log_param(
                                    f"{self.model_trainer_config.UTILS.sanitize_key(k)}", str(v)
                                )


                    logging.info(f"All models logged successfully to MLflow with run ID: {mlflow.active_run().info.run_id}")
                    
                # ------------------------------------------------------------------------------------------------------------ 
                self.model_trainer_config.UTILS.update_model_score(best_model_score)

                logging.info(f"Model logged to MLflow & saved at {model_file_path}")
                model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=model_file_path)
                logging.info(f"Model Trainer Artifacts: {model_trainer_artifacts}")

                return model_trainer_artifacts
            
            else:
                logging.info("No better model found. Keeping previous best model artifact unchanged.")

                existing_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info(f"Using existing model at path: {existing_model_path}")

                if not os.path.exists(existing_model_path):
                    logging.warning("No previous model found. Returning artifacts with empty path.")
                    return Model_Trainer_Artifacts(trained_model_file_path=None)

                model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=existing_model_path)
                logging.info(f"Returning existing model artifact: {model_trainer_artifacts}")
                return model_trainer_artifacts

            
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(str(e), sys)




if __name__ == "__main__":
    try:
        load_dotenv()

        # Set username & token for DagsHub auth
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

        # Set URI via dagshub or manually
        dagshub.init(repo_owner='rashmiKumari03', repo_name='Shipment-Price-Prediction', mlflow=True)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        logging.info(f"MLFLOW URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        mlflow.set_experiment("Shipment_Price_Prediction_Tracking")
        
        print("USERNAME:", os.getenv("MLFLOW_TRACKING_USERNAME"))
        print("TOKEN exists:", bool(os.getenv("MLFLOW_TRACKING_PASSWORD")))


        logging.info("*******************")
        logging.info(">>>>>> Model Trainer stage started <<<<<<")

        # ------------------------------------------------------
        # Creating the Data Transformation Artifacts object
        # ------------------------------------------------------
        data_transformation_config = Data_Transformation_Config()
        data_transformation_artifact = Data_Transformation_Artifacts(
            train_file_path= data_transformation_config.TRAIN_FILE_PATH ,
            test_file_path= data_transformation_config.TEST_FILE_PATH,
            transformed_object_file_path= data_transformation_config.PREPROCESSOR_FILE_PATH
        )

        # ------------------------------------------------------
        # Creating the Model Trainer Config
        # ------------------------------------------------------
        model_trainer_config = Model_Trainer_Config()

        # ------------------------------------------------------
        # Instantiating the Model_Trainer class
        # ------------------------------------------------------
        model_trainer = Model_Trainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config
        )

        # ------------------------------------------------------
        # Calling the Model Trainer pipeline
        # ------------------------------------------------------
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info(">>>>>> Model Trainer stage completed <<<<<<\n")
        logging.info(f"Model Trainer Artifacts: {model_trainer_artifact}")

    except Exception as e:
        logging.exception(e)
        raise e
