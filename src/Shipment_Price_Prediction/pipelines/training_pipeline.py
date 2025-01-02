import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.configuration.mongo_operation import MongoDB_Operation
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts)
from src.Shipment_Price_Prediction.entity.config_entity import (Data_Ingestion_Config)

from src.Shipment_Price_Prediction.components.data_ingestion import Data_Ingestion


# Initializing the Training Pipeline.
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_Config()
        self.mongo_op = MongoDB_Operation()
        
    
    # The method is used to start the data ingestion
    def start_data_ingestion(self) -> Data_Ingestion_Artifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        
        try:
            logging.info("Getting the data from MongoDB")
            data_ingestion = Data_Ingestion(data_ingestion_config= self.data_ingestion_config , mongo_op=self.mongo_op)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            
            return data_ingestion_artifact
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
    # To start this data ingestion we need to make another method call run_pipeline
    # This method is used to start the training pipeline
    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of Training class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
                
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
    # Lets start the data ingestion pipeline.. for that we can go to trial_runner_app.py 
