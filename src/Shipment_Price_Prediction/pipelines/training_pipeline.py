import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.configuration.mongo_operation import MongoDB_Operation
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts, Data_Validation_Artifacts,Data_Transformation_Artifacts,Model_Trainer_Artifacts,Model_Evaluation_Artifacts)
from src.Shipment_Price_Prediction.entity.config_entity import (Data_Ingestion_Config , Data_Validation_Config,Data_Transformation_Config,Model_Trainer_Config,Model_Evaluation_Config)

from src.Shipment_Price_Prediction.components.data_ingestion import Data_Ingestion
from src.Shipment_Price_Prediction.components.data_validation import Data_Validation
from src.Shipment_Price_Prediction.components.data_transformation import Data_Transformation
from src.Shipment_Price_Prediction.components.model_trainer import Model_Trainer
from src.Shipment_Price_Prediction.components.model_evaluation import Model_Evaluation
from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation


# Initializing the Training Pipeline.
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_Config()
        self.mongo_op = MongoDB_Operation()
        
        self.data_validation_config = Data_Validation_Config()
        self.data_transformation_config = Data_Transformation_Config()
        self.model_trainer_config = Model_Trainer_Config()
        self.model_evaluation_config = Model_Evaluation_Config()
        
        # Also need to initialise S3
        self.s3_operations = S3_Operation()
        
        
    
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
        
        
    
    # IMPORTING : Adding Data_Validation_Artifacts and Data_Validation_Config and from components.data_validation import Data_Validation class....
    # in __int__ call the self.data_validation_config = Data_Validation_Config()
    # Now Initializing the method start_data_validation()
    
    # This method is used to start the data validation
    def start_data_validation(self,data_ingestion_artifact: Data_Ingestion_Artifacts) -> Data_Validation_Artifacts:
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = Data_Validation(
                data_ingestion_artifacts= data_ingestion_artifact,
                data_validation_config= self.data_validation_config
            )
            
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            
            return data_validation_artifact
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
        
        
        
    # This method is used to start the data transformation
    def start_data_transformation(self,data_ingestion_artifact : Data_Ingestion_Artifacts) -> Data_Transformation_Artifacts:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = Data_Transformation(
                data_ingestion_artifacts= data_ingestion_artifact,
                data_transformation_config= self.data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainPipeline")
            
            return data_transformation_artifact
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
    
    
        
    # Starting the model training.. it will take models from model.yaml one by one and train the data with those model and compare to get the best model 
    def start_model_trainer(self,data_transformation_artifact : Data_Transformation_Artifacts) -> Model_Trainer_Artifacts:
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline")
            model_trainer = Model_Trainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config= self.model_trainer_config
                )
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline")
            return model_trainer_artifact
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
    # Starting the model evaluation ...
    def start_model_evaluation(self,data_ingestion_artifact : Data_Ingestion_Artifacts , model_trainer_artifact : Model_Trainer_Artifacts ) -> Model_Evaluation_Artifacts:
        try:
            model_evaluation = Model_Evaluation( model_evaluation_config = self.model_evaluation_config,
                                                data_ingestion_artifact = data_ingestion_artifact,
                                                model_trainer_artifact= model_trainer_artifact
                                                )
        except Exception
    
    
        
        
    
   
    # To start this data ingestion,validation etc... we need to make another method call run_pipeline
    # This method is used to start the training pipeline
    
    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of Training class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact= data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("Exited the run_pipeline method of TrainPipeline class")
                
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
            
    # Lets start the pipeline.. for that we can go to trial_runner_app.py
    