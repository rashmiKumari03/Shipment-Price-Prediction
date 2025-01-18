import os 
import sys
import pandas as pd
from dataclasses import dataclass

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts,Model_Trainer_Artifacts,Model_Evaluation_Artifacts)
from src.Shipment_Price_Prediction.entity.config_entity import Model_Evaluation_Config

# Define constants for model evaluation in the constants folder.
# Implement the S3 bucket configuration in the configuration folder for better organization.

# This class elements were logged.
@dataclass
class Evaluate_Model_Response:
    trained_model_r2_score: float
    s3_model_r2_score: float
    is_model_accepted: bool
    difference: float
    
    
class Model_Evaluation:
    def __init__(self,model_trainer_artifact : Model_Trainer_Artifacts,model_evaluation_config : Model_Evaluation_Config,data_ingestion_artifact : Data_Ingestion_Artifacts):
        
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        
        
        
    # This method is used to get the s3 model
    def get_s3_model(self) -> object:
        
        """ 
        Method Name : get_s3_model
        
        Description : This method gets model from s3 bucket.
        
        Output      : Model
    
        """
        logging.info("Entered the get_s3_model method od the Model_Evaluation class")
        try:
            # Checking whether model is present in the S3 bucket or not?
            status = self.model_evaluation_config.S3_Operation.is_model_present(BUCKET_NAME, S3_MODEL_NAME)
            logging.info(f"Got the status : is model present ? => {status}")
            
            # If model is present then loading the model
            if status == True:
                model = self.model_evaluation_config.S3_Operation.load_model(MODEL_FILE_NAME,BUCKET_NAME)
                logging.info("Exited the get_s3_model method of Model_Evaluation class")
                
                return model
            else:
                logging.info("Model Not Found !")
                
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)

            
        
    