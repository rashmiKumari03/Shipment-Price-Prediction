import os
import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import List,Tuple

from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE 
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
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
            logging.info(f"X looks like : {X}")
            transformed_feature = self.preprocessing_object.transform(X)
            logging.info("Transformed features using preprocessing object")

            predictions = self.trained_model_object.predict(transformed_feature)
            logging.info("Used the trained model to get predictions")
            return predictions

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
            
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            logging.info("Model_List")
            
            models_list = list(model_config.get("train_model", {}).keys())
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
       # This method is used to initialize model training
    def initiate_model_trainer(self) -> Model_Trainer_Artifacts:
        """
        Model Name  : initiate_model_trainer
        
        Description : This method initiates the model training process by:
                      - Loading transformed training and test data
                      - Training multiple models and selecting the best one
                      - Comparing the best model's score with a base model score
                      - Saving the best model if its score exceeds the base score
                      - Creating and returning model trainer artifacts
        
        Output      : Model_Trainer_Artifacts containing the path to the saved model
        """
        logging.info("Entered initiate_model_trainer method of Model_Trainer class")
        try:
            # Ensure the artifacts directory exists, where trained models and logs will be stored
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.model_trainer_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Load the transformed training data from the pre-saved numpy array
            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            logging.info(f"Data type of training_array is {type(train_array)}")
            logging.info(f"Training array :\n{train_array}")
            logging.info(f"Training array size :\n{train_array.shape}")
                      
            # Load the transformed test data from the pre-saved numpy array
            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Data type of testing_array is {type(test_array)}")
            logging.info(f"Testing array : \n{test_array}")
            logging.info(f"Testing array size : \n{test_array.shape}")
            logging.info("-"*120)
            
            # Convert numpy arrays into pandas DataFrames for easier manipulation during model training
            train_df = pd.DataFrame(train_array)
            test_df = pd.DataFrame(test_array)          
        
            logging.info(f"Converted train and test arrays to DataFrames: Train shape {train_df.shape}, Test shape {test_df.shape}")
    
            # Train multiple models and get a list of (model score, model object, model name)
            list_of_trained_models = self.get_trained_models(train_df, test_df)
            logging.info("Got a list of tuple of model score, model, and model name")
            
            # Identify the best model, its score, and its name
            best_model_name , best_model_score , best_model_object = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(list_of_trained_models)

            # Log the best model details
            logging.info(f"Best model: {best_model_name}, Model object: {best_model_object}, Score: {best_model_score}")
            logging.info("Successfully identified the best model, its score, and name.")
            
            # Load the preprocessor object used during data transformation
            preprocessor_obj_file_path = self.data_transformation_artifact.transformed_object_file_path
            logging.info(f"The Preprocessor_obj_file_path is : {preprocessor_obj_file_path}")
            
            preprocessor_obj = self.model_trainer_config.UTILS.load_object(preprocessor_obj_file_path)
            logging.info(f"Preprocessor object is: {preprocessor_obj}")
            
            logging.info("Loaded preprocessing object")

            # Read the base model score from the model config YAML file
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            base_model_score = float(model_config["base_model_score"])
            logging.info(f"Base model score from config: {base_model_score}")
            logging.info(f"Best model score : {best_model_score}")
        

            # Compare the best model's score with the base model score
            # Save the model only if the best model's score is greater than or equal to the base score
            if float(best_model_score) >= float(base_model_score):
                
                # Update the base model score in the YAML file
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                logging.info("Updating the model score in yaml file")

                # Create a cost model object containing the preprocessor and the best model
                cost_model = Cost_Model(preprocessor_obj, best_model_object)
            
                logging.info("Created cost model object with preprocessor and model")
                
                # Define the path for saving the trained model
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info("Created best model file path")
                logging.info(f"Best model path is :{trained_model_path}")
            

                # Save the trained model to the specified path
                model_file_path = self.model_trainer_config.UTILS.save_object(trained_model_path, cost_model)
                logging.info(f"Saved the best model object at {model_file_path}")
                
            else:
                # Log a message if the best model doesn't surpass the base score
                logging.info("No best model found with score higher than the base model score")
                raise "NO best model found with the score more than the base score"
                
    
            # Save the model trainer artifacts, including the trained model's file path
            model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=model_file_path)
            
            logging.info(f"Model trainer artifacts saved at {model_trainer_artifacts.trained_model_file_path}")
        
            # Return the artifacts for further processing
            return model_trainer_artifacts
        
        except Exception as e:
            # Handle and log any exceptions that occur during model training
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)