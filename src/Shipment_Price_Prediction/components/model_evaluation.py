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
                None
                
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    # This method is used to evaluate the model
    
    def evaluate_model(self) -> Evaluate_Model_Response:
        """ 
        Method Name : evaluate_model 
        
        Description : This method evaluates s3 bucket model and production model
        
        Output      : This model give output about the evaluation metric , whether model is accepted or not
        """
        logging.info("Entered the evaluate_model method of Model evaluation class")
        try:
            # Reading the test data and splitting it into train and test
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            
            # Issue hai yaha test_df me Shipment_Price target col defined nahi hai...then how to do it
            x,y = test_df.drop(TARGET_COLUMN,axis=1),test_df[TARGET_COLUMN]
            logging.info("splitted the test data into train and test")
            
            # Loading production model for prediction
            trained_model = self.model_evaluation_config.UTILS.load_object(self.model_trainer_artifact.trained_model_file_path)
            y_hat_trained_model = trained_model.predict(x)
            logging.info("Prediction done with production model")
            
            # Checking the r2 score of production model
            trained_model_r2_score =
            
            
            

            
        
    