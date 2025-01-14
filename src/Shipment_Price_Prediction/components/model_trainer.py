import os
import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
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
        
    def predict(self,X)-> float:
        """ 
        Method Name : predict
        
        Description : This method predicts the data
        
        Output      : Predictions
        """
        logging.info("Entered predict method of the Cost_Model class")
        try:
            # Using the trained model tp get predcition
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
    def __init__(self,data_transformation_artifact:Data_Transformation_Artifacts,model_trainer_config:Model_Trainer_Config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        
    # This method is used to get the trained models
    def get_trained_models(self,X_data:DataFrame,y_data:DataFrame)-> List[Tuple[float,object,str]]:
        """ 
        Method Name : get_trained_models
        
        Description : This method lists the trained model
        
        Output      : List of trained models
        
        """
        logging.info("Entered get_trained_models method of Model_Trainer class")
        
        try:
            # Getting the model lists from model config file
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            models_list = list(model_config["train_model"].keys())
            logging.info("Got model list from the config file")
            
            
            # Splitting the data into x_train , x_test , y_train and y_test
            x_train , x_test , y_train , y_test = (X_data.drop(X_data.columns[len(X_data.columns)-1],axis=1),
                                                   X_data.iloc[:,-1],
                                                   y_data.drop(y_data.columns[len(y_data.columns)-1],axis=1),
                                                   y_data.iloc[:,-1])
            
            # Getting the trained model list
            tunned_model_list = [
                (
                    self.model_trainer_config.UTILS.get_tunned_model(model_name,x_train,y_train,x_test,y_test)
                )
                for model_name in models_list
            ]
            logging.info("got trained model list")
            logging.info("Exited the get_trained_models method of Model_Trainer class")
            
            return tunned_model_list
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    def initiate_model_trainer(self) -> Model_Trainer_Artifacts:
        
        """
        Model Name  : initiate_model_trainer
        
        Description : This method initiates model training
        
        Output      :  List of trained models
        
        """
        logging.info("Entered initiate_model_trainer method of Model_Trainer class")
        try:
            
            # Creating Model trainer artifacts directory
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,exist_ok=True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.model_trainer_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Loading the train array data and reading it as DataFrame
            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            train_df = pd.DataFrame(train_array)
            logging.info(f"Loaded train array from Data_Transformation_Artifacts directory and converted into DataFrame")
            
            # Loading the test array data and reading it as DataFrame
            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            test_df = pd.DataFrame(test_array)
            logging.info(f"Loaded test array from Data_Transformation_Artifacts directory and converted into DataFrame")
            
            # Getting the models list and finding the best model with score
            list_of_trained_models = self.get_trained_models(train_df,test_df)
            logging.info("Got a list of tuple of model score , model and model name")
            best_model , best_model_score = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(list_of_trained_models)
            logging.info("Got best model score , model and model name")
            
            # Loading the preprocessor object
            preprocessor_obj_file_path = self.data_transformation_artifact.transformed_object_file_path
            preprocessor_obj = self.model_trainer_config.UTILS.load_object(preprocessor_obj_file_path)
            logging.info("Loaded preprocessing object")
            
            # Reading the model config file for getting the best model score
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            base_model_score = float(model_config['base_model_score'])
            
            
            # Updating the model score
            if best_model_score >= base_model_score: 
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                logging.info("Updating the model score in yaml file")
                
                # Loading the cost model object with preprocessor and model
                cost_model = Cost_Model(preprocessor_obj , best_model)
                logging.info("Created cost model object with preprocessor and model")
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                
                # Saving cost model in model artifacts directory
                model_file_path = self.model_trainer_config.UTILS.save_object(trained_model_path,cost_model)
                logging.info("Saved the best model object path")
            else:
                logging.info("No best model found with score more than base score")
                raise "No best model found with score more than base score"
            
            
            # Saving the model trainer artifacts
            model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path= model_file_path)
            
            return model_trainer_artifacts
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
           
        
