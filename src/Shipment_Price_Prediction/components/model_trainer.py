import os
import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import numpy as np
import pandas as pd


from pandas import DataFrame
from typing import List,Tuple

from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE 
from src.Shipment_Price_Prediction.entity.config_entity import Model_Trainer_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Transformation_Artifacts,Model_Trainer_Artifacts)

class Cost_Model:
    
    def __init__(self,preprocessing_object: object , trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        
    def predict(self, X)-> float:
        """ 
        Method Name : predict
        
        Description : This method predicts the data
        
        Output      : Predictions
        """
        logging.info("Entered predict method of the Cost_Model class")
        try:
            # Using the trained model to get predcitions
            transformed_feature = self.preprocessing_object.transform(X)
            logging.info("Used the trained model to get predictions")
            
            return self.trained_model_object.predict(transformed_feature)
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    
    
class Model_Trainer:
    
    def __init__(self,
                 data_transformation_artifact:Data_Transformation_Artifacts,
                 model_trainer_config:Model_Trainer_Config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
    
        
    # This method is used to get the trained models
    def get_trained_models(self, train_data : DataFrame , test_data : DataFrame)-> List[Tuple[float,object,str]]:
        """ 
        Method Name : get_trained_models
        
        Description : This method lists the trained model
        
        Output      : List of trained models
        
        """
        logging.info("Entered get_trained_models method of Model_Trainer class")
        
        try:
            
            model_config = self.model_trainer_config.SCHEMA_CONFIG
            logging.info("Model_List")
            models_list = list(model_config["train_model"].keys())
            logging.info(models_list)
            logging.info("Fetched models from configuration file")

            # Splitting the data into  x_train , y_train and x_test , y_test
            x_train,y_train,x_test, y_test = (train_data.drop(train_data.columns[len(train_data.columns) - 1],axis=1),
                                              train_data.iloc[:,-1],
                                              test_data.drop(test_data.columns[len(test_data.columns)- 1],axis=1),
                                              test_data.iloc[:,-1]
                                              )
            
            logging.info(f"x_train looks like:{x_train.head()}")
            logging.info(f"x_test looks like:{x_test.head()}")
            logging.info(f"y_train looks like:{y_train.head()}")
            logging.info(f"y_test looks like:{y_test.head()}")
                                         
            # Getting the trained model list
            tunned_model_list = [
                self.model_trainer_config.UTILS.get_tunned_model(model_name,x_train,y_train,x_test,y_test)
                for model_name in models_list
            ]
            logging.info("got trained model list")
            logging.info("Exited the get_trained_models method of Model_Trainer class")
            
            return tunned_model_list
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)

        
        
    # This method is used to initialize model training
    def initiate_model_trainer(self) -> Model_Trainer_Artifacts:
    
        
        """
        Model Name  : initiate_model_trainer
        
        Description : This method initiates model training
        
        Output      :  List of trained models
        
        """
        logging.info("Entered initiate_model_trainer method of Model_Trainer class")
        try:
            # Creating Model trainer artifacts directory
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.model_trainer_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Loading the train array data and reading it as DataFrame
            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            logging.info(f"Data type of training_array is {type(train_array)}")
            logging.info(f"Training array :\n{train_array}")
            logging.info(f"Training array size :\n{train_array.shape}")
                      
            # Loading the test array data and reading it as DataFrame
            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Data type of testing_array is {type(test_array)}")
            logging.info(f"Testing array : \n{test_array}")
            logging.info(f"Testing array size : \n{test_array.shape}")
            logging.info("-"*120)
            
            # Coversion of array into Datarame
            train_df = pd.DataFrame(train_array)
            test_df = pd.DataFrame(test_array)          
        
        
            logging.info(f"Converted train and test arrays to DataFrames: Train shape {train_df.shape}, Test shape {test_df.shape}")
    

            # Getting the models list and finding the best model with score
            list_of_trained_models = self.get_trained_models(train_df, test_df)
            logging.info("Got a list of tuple of model score, model, and model name")
            
            # Finding the best model, its name, and score
            best_model_name , best_model_score , best_model_object = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(list_of_trained_models)

            # Logging the details of the best model
            logging.info(f"Best model: {best_model_name}, Model object: {best_model_object}, Score: {best_model_score}")
            logging.info("Successfully identified the best model, its score, and name.")

            # Loading the preprocessor object
            preprocessor_obj_file_path = self.data_transformation_artifact.transformed_object_file_path
            logging.info(f"The Preprocessor_obj_file_path is : {preprocessor_obj_file_path}")
            
            preprocessor_obj = self.model_trainer_config.UTILS.load_object(preprocessor_obj_file_path)
            logging.info(f"Preprocessor object is: {preprocessor_obj}")
            
            logging.info("Loaded preprocessing object")

            # Reading the model config file for getting the base model score
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            
            base_model_score = float(model_config["base_model_score"])
            logging.info(f"Base model score from config: {base_model_score}")
            
            # Setting a default value for model_file_path before the comparison
            model_file_path = None

            # Updating the model score if the best model is better than the base model score
            
            """
            WARNING: 

            If the best_model_score is lower than the base_model_score, the model will not be saved.
            Ensure that the base_model_score in model.yaml is initialized to a reasonable value (e.g., 0.1)
            before starting the training process. 

            If the base_model_score is set too high or not properly configured, it could prevent the best model 
            from being saved, even if the model performs well. This may result in missing the best model due to 
            an incorrectly configured score threshold.

            Make sure to double-check the base_model_score value before running the training to avoid potential issues.
            
            """


            if float(best_model_score) >= float(base_model_score):
                
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                logging.info("Updating the model score in yaml file")

                # Loading the cost model object with preprocessor and model
                cost_model = Cost_Model(preprocessor_obj, best_model_object)
                logging.info("Created cost model object with preprocessor and model")
                
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info("Created best model file path")

                # Saving the trained model in the model artifacts directory
                model_file_path = self.model_trainer_config.UTILS.save_object(trained_model_path, cost_model)
                logging.info(f"Saved the best model object at {model_file_path}")
                
            else:
                logging.info("No best model found with score higher than the base model score")
                
            
            # If no model was saved, you can handle it by setting model_file_path to a default or None
            if model_file_path is None:
                logging.error("Model was not saved due to insufficient score.")
                return None  # or raise a CustomException if needed

            # Saving the model trainer artifacts
            model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=model_file_path)
            logging.info(f"Model trainer artifacts saved at {model_trainer_artifacts.trained_model_file_path}")

            return model_trainer_artifacts
        
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)          
        
