import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Tuple

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE, SCHEMA_FILE_PATH
from src.Shipment_Price_Prediction.components.data_transformation import Data_Transformation
from src.Shipment_Price_Prediction.entity.config_entity import Model_Trainer_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import Data_Transformation_Artifacts, Model_Trainer_Artifacts

class Cost_Model:

    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.model = trained_model_object
        

    def prediction(self, X:DataFrame):
        """
        Predict shipment costs based on input features.
        """
        logging.info("Entered the 'prediction' method of the Cost_Model class.")
        try:
            logging.info(f"X looks like : \n {X.head()}")
            # Preprocess the input data
            
            # Loading the preprocessor.pkl
            logging.info(f"Preprocessor and Model object provided in the predict..")
            
            logging.info(f"preprocessor_obj is :{self.preprocessing_object}")
            X_transformed = X.drop_duplicates()
            X_transformed = self.preprocessing_object.transform(X)
            # Make predictions using the trained model
            logging.info("Transformation on new/test data has Completed!")
            return self.model.predict(X_transformed)
            

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(str(e), sys)

    def __repr__(self):
        return f"{type(self.model).__name__}()"

    def __str__(self):
        return f"{type(self.model).__name__}()"


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
            
            best_model_name, best_model_score, best_model_object = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(trained_models)
            
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            base_model_score = float(model_config["base_model_score"])

            if best_model_score >= base_model_score:
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                
                preprocessor_obj = self.model_trainer_config.UTILS.load_object(self.data_transformation_artifact.transformed_object_file_path)
                logging.info("Here i am Saving the cost_model using save_object!!!!")
                cost_model = Cost_Model(preprocessor_obj, best_model_object)
                logging.info(f"cost_model looks like : {cost_model} and type of cost_model is {type(cost_model)}")
                model_file_path = self.model_trainer_config.UTILS.save_object(self.model_trainer_config.TRAINED_MODEL_FILE_PATH, cost_model)
                logging.info(f"model_file_path looks like : {model_file_path}")
                model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=model_file_path)
                logging.info(f"Model saved at {model_file_path}")
                logging.info(f"Model Trainer Artifacts : {model_trainer_artifacts}")

                return model_trainer_artifacts
            
            else:
                # Save base model again if no better model is found
                logging.warning(f"No model exceeded the base model score of {base_model_score}. Best model score: {best_model_score}.")
                
                # Load the existing base model
                base_model_object = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
                
                # Define the new path to save the model in Model_Trainer_Artifacts
                model_file_path = os.path.join(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, "Retained_Model.pkl")
                
                # Save the base model directly without wrapping it
                self.model_trainer_config.UTILS.save_object(model_file_path, base_model_object)
                logging.info(f"Base model re-saved at {model_file_path}.")
                
                model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=model_file_path)
                return model_trainer_artifacts
            
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(str(e), sys)
