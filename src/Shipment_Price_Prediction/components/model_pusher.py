import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Transformation_Artifacts , Model_Trainer_Artifacts , Model_Pusher_Artifacts)

from src.Shipment_Price_Prediction.entity.config_entity import Model_Pusher_Config

# Lets  Create first the Model_Pusher_Artifacts and Model_Pusher_Config

class Model_Pusher:
    
    def __init__(self, 
                 model_pusher_config : Model_Pusher_Config ,
                 model_trainer_artifacts : Model_Trainer_Artifacts , 
                 data_transformation_artifacts: Data_Transformation_Artifacts,
                 s3: S3_Operation
                 ):
        
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.s3 = s3
        
        
    # This is method is used to initiate the model pusher.
    def initiate_model_pusher(self) -> Model_Pusher_Artifacts:
        """ 
        Method Name : initiate_model_pusher
        
        Description : This method initiates model pusher.
        
        Output      : Model Pusher Artifacts
        """
        logging.info("Entered the initiate_model_pusher method of Model_Pusher class")
        try:
            # Uploading the best model to S3 Bucket
            
            self.s3.upload_file(self.model_trainer_artifacts.trained_model_file_path,
                                self.model_pusher_config.S3_MODEL_KEY_PATH,
                                self.model_pusher_config.BUCKET_NAME,
                                remove=False
                                )
            logging.info("Uploaded the best model to S3 Bucket")
            logging.info("Exited the initiate_model_pusher method of Model_Pusher class")
            
            # Saving the model pusher artifacts:
            model_pusher_artifact = Model_Pusher_Artifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH)
            
            return model_pusher_artifact
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
            
                         
        
    
