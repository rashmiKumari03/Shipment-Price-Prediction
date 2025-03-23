from dataclasses import dataclass
from from_root import from_root
import os

from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation

from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
from src.Shipment_Price_Prediction.constant import * 


# class Data_Ingestion_Config: Configuration class for the data ingestion process.

@dataclass
class Data_Ingestion_Config:
    
    def __init__(self):
        
        """
        Configuration class for the data ingestion process.
        This class sets up paths and parameters required for ingesting data.
        """
        
    
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



# class Data_Validation_Config : Configuration class for the data validation process.

@dataclass
class Data_Validation_Config:
    """
    Configuration class for the data validation process.
    This class provides paths and settings required for validating the data.
    """

    def __init__(self):
        # Initialize utility class for common operations
        self.UTILS = MainUtils()

    
        # This class returns all the necessary paths required in Data Validation 
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)

    
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_VALIDATION_ARTIFACTS_DIR: str = os.path.join(from_root(),ARTIFACTS_DIR , DATA_VALIDATION_ARTIFACT_DIR)
        self.DATA_DRIFT_FILE_PATH: str = os.path.join(self.DATA_VALIDATION_ARTIFACTS_DIR, DATA_DRIFT_FILE_NAME)
        
        
# class Data_Transformation_Config : Configuration class for the data transformation process.
@dataclass
class Data_Transformation_Config :
    def __init__(self):
        
        self.UTILS = MainUtils()
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
        
        self.DATA_INGESTION_ARTIFACTS_DIR:str = os.path.join(from_root(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR:str = os.path.join(from_root(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)

        
        
        # Define file paths for transformed data
        self.TRAIN_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRAIN_DATA_FILE_NAME)
        self.TEST_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TEST_DATA_FILE_NAME)
        self.PREPROCESSOR_FILE_PATH:str = os.path.join(from_root(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR,PREPROCESSOR_OBJECT_FILE_NAME)
        
        
# class Model_Trainer_Config :  Configuration class for model training, handling paths and settings.
@dataclass
class Model_Trainer_Config:
    
    def __init__(self):
        self.UTILS = MainUtils()
        
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(from_root(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(from_root(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.PREPROCESSOR_OBJECT_FILE_PATH:str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,PREPROCESSOR_OBJECT_FILE_NAME)
        self.TRAINED_MODEL_FILE_PATH:str = os.path.join(from_root(),ARTIFACTS_DIR ,MODEL_TRAINER_ARTIFACTS_DIR,MODEL_FILE_NAME)
        
        
# Import the S3 bucket functionality and write the S3 operations code in the configuration folder for modularity.

# Model Evaluation Configuration
@dataclass
class Model_Evaluation_Config:
    
    def __init__(self):
        self.S3_Operation = S3_Operation()
        self.UTILS = MainUtils()
        self.BUCKET_NAME : str = BUCKET_NAME
        self.BEST_MODEL_PATH : str = os.path.join(from_root(), ARTIFACTS_DIR , MODEL_TRAINER_ARTIFACTS_DIR , MODEL_FILE_NAME)
        
 
 
# Model_Pusher_Config for saving the best trained model (which is saved locally) , to upload it in the S3 bucket of AWS.       
@dataclass
class Model_Pusher_Config:
    
    """
    This class is used to define the configuration for the Model Pusher component, which is responsible
    for saving the best-trained model locally and uploading it to an S3 bucket for further use.
    """
    
    def __init__(self):
        
         # Path to the best-trained model saved locally
        self.BEST_MODEL_PATH: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, MODEL_FILE_NAME)
        
        # Name of the S3 bucket where the model will be uploaded
        self.BUCKET_NAME: str = BUCKET_NAME
        
        # S3 path (key) where the model file will be stored in the specified bucket
        self.S3_MODEL_KEY_PATH: str = os.path.join(S3_MODEL_NAME)