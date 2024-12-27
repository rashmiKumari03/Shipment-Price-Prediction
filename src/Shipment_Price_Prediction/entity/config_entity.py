from dataclasses import dataclass
from from_root import from_root
import os
from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
from src.Shipment_Price_Prediction.constant import * 

@dataclass
class Data_Ingestion_Config:
    
    def __init__(self):
        # This will help us use common functions (like reading a YAML file) later.
        self.UTILS = MainUtils()

        # This reads the schema configuration from the YAML file.
        # The schema file tells us which columns to keep or drop, and other important settings.
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
        
        # These two variables store the database name and collection name where our data is saved.
        self.DB_NAME = DB_NAME  # The name of the database.
        self.COLLECTION_NAME = COLLECTION_NAME  # The name of the collection inside the database.

        # This reads the columns to be dropped, as specified in the schema file.
        # For example, if there are columns that are not needed, this will remove them.
        self.DROP_COLS = list(self.SCHEMA_CONFIG["drop_columns"])  


        # Define the main directory where all artifacts (processed data) will be stored
        # This is where the processed data (artifacts) will be saved. 
        # It's like creating a folder to keep everything organized.
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR
        )
        
        
        
        # Define the subdirectories where training and testing data will be stored
        # These are the paths where we will save the training and testing data separately.
        # The training data will be used to teach the model, and the testing data will be used to check how good the model is.
        self.TRAIN_DATA_ARTIFACT_FILE_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAIN_DIR
        )
        self.TEST_DATA_ARTIFACT_FILE_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TEST_DIR
        )
        
        
        
        # Define the final file paths for saving the training and test datasets
        # These are the actual file paths for saving the training and testing datasets.
        # When the data is ready, we will save it to these files.
        self.TRAIN_DATA_FILE_PATH: str = os.path.join(
            self.TRAIN_DATA_ARTIFACT_FILE_DIR, DATA_INGESTION_TRAIN_FILE_NAME
        )
        self.TEST_DATA_FILE_PATH: str = os.path.join(
            self.TEST_DATA_ARTIFACT_FILE_DIR, DATA_INGESTION_TEST_FILE_NAME
        )
        
# After this all , now creating artifacts_entity.py...