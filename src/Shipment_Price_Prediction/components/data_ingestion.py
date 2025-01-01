
import sys
import os

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.Shipment_Price_Prediction.configuration.mongo_operation import MongoDB_Operation

from src.Shipment_Price_Prediction.entity.config_entity import Data_Ingestion_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import Data_Ingestion_Artifacts

from src.Shipment_Price_Prediction.constant import TEST_SIZE

# Note : We first have to create Data_Ingestion_Config , Data_Ingestion_Artifacts and TEST_SIZE in their respective path.
"""
Data Ingestion Flow Overview:

1. Data_Ingestion_Config:
   - Inputs:
     - data_ingestion_dir: Directory for storing ingestion artifacts.
     - feature_store_file_path: Path to store the feature store data.
     - training_file_path: Path for training data file.
     - testing_file_path: Path for testing data file.
     - train_test_split_ratio: Ratio for splitting data into train and test sets.
     - collection_name: MongoDB collection name for source data.

2. Initiate_Data_Ingestion:
   - Orchestrates the complete data ingestion workflow.

3. Steps in Data Ingestion:

   a) Export_Data_to_Feature_Store:
      - Input:
        - Connects to MongoDB using collection_name.
        - Retrieves raw data from the source.
      - Output:
        - Stores the raw dataset in feature_store_file_path under Data_Ingestion_Artifact.

   b) Drop_Columns:
      - Input:
        - Reads the feature store dataset.
        - Uses a schema file to identify columns to drop.
      - Output:
        - Saves the cleaned dataset back to the feature store.

   c) Split_Data_as_train_and_test:
      - Input:
        - Reads the cleaned dataset from the feature store.
        - Splits the data into train and test sets based on train_test_split_ratio.
      - Output:
        - Stores train_data and test_data as train.csv and test.csv under Data_Ingestion_Artifact.

4. Final Outputs:
   - Feature_Store: Contains the complete cleaned dataset.
   - Train_Test_Split: Train and test datasets saved as:
     - Data_Ingestion_Artifact/timestamp/ingested/train.csv
     - Data_Ingestion_Artifact/timestamp/ingested/test.csv

Detailed Flow Diagram:
    ┌──────────────────────────────┐
    │     Data_Ingestion_Config    │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │   Initiate_Data_Ingestion    │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │ Export_Data_to_Feature_Store │
    │  - Connect to MongoDB        │
    │  - Retrieve raw data         │
    │  - Save to feature store     │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │         Drop_Columns         │
    │  - Load schema file          │
    │  - Drop unwanted columns     │
    │  - Save cleaned dataset      │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │ Split_Data_as_train_and_test │
    │  - Load cleaned dataset      │
    │  - Split into train/test sets│
    │  - Save train.csv, test.csv  │
    └──────────────┬───────────────┘
        ┌──────────┴─────────────┐
        ▼                        ▼
    train.csv                 test.csv
      │                          │
      ▼                          ▼
Data_Ingestion_Artifact/    Data_Ingestion_Artifact/
timestamp/ingested/         timestamp/ingested/

Purpose:
This workflow ensures modular, reproducible, and efficient data ingestion for machine learning pipelines, with clear separation of raw data, cleaned data, and training/testing datasets.
"""

# Creating data ingestion related constants in constant folder

class Data_Ingestion:
    
    def __init__(self,data_ingestion_config : Data_Ingestion_Config , mongo_op : MongoDB_Operation):
        
        self.data_ingestion_config = data_ingestion_config
        self.mongo_op = mongo_op
        
    
    # This method will fetch data from mongoDB
    def get_data_from_mongodb(self) -> DataFrame:
        
        """
        Method Name : get_data_from_mongodb
        
        Description : This method fetches data from MongoDB database
        
        Output : DataFrame
        
        """
        logging.info("Entered get_data_from_mongodb method of Data_Ingestion class")
    
        try:
            logging.info("Getting the dataframe from mongodb")
            
            # Getting collection from MongoDB database
            df = self.mongo_op.get_collection_as_dataframe(self.data_ingestion_config.DB_NAME,
                                                            self.data_ingestion_config.COLLECTION_NAME)
            logging.info("Got the dataframe from mongodb")
            logging.info("Exited the get_data_from_mongodb method of Data_Ingestion class")
            
            return df
        
        except Exception as e :
            raise CustomException(e,sys)
        
        
        
    # This method will split the data
    def split_data_as_train_test(self,df : DataFrame) -> Tuple[DataFrame,DataFrame]:
        
        """
            Method Name : split_data_as_train_test
            
            Description : This method splits the dataframe into trainset and testset based on the split ratio.
            
            Output : Train DataFrame as Test DataFrame
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            
            # Creating Data Ingestion Artifacts directory inside Artifact folder
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,exist_ok=True)
            
            
            # Splitting the data into train and test
            train_set , test_set =  train_test_split(df , test_size = TEST_SIZE)
            logging.info("Performed train test split on the dataframe")
            
            # Creating train directory under data ingestion artifact directory
            os.makedirs(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR,exist_ok=True)
            logging.info(f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR)} directory")
            
            # Creating test directory under data ingestion artifact directory
            os.makedirs(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR,exist_ok=True)
            logging.info(f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR)} directory")
            
            
            # Saving train.csv file to train directory
            train_set.to_csv(self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                             index = False,
                             header = True)
            
            # Saving test.csv file to test directory
            test_set.to_csv(self.data_ingestion_config.TEST_DATA_FILE_PATH,
                            Index = False,
                            header = True)
            
            logging.info("Converted Train DataFrame and Test DataFrame into csv")
            logging.info(f"Saved {os.path.basename(self.data_ingestion_config.TRAIN_DATA_FILE_PATH)},\
                {os.path.basename(self.data_ingestion_config.TEST_DATA_FILE_PATH)} in\
                {os.path.basename(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)}.")
            
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            
            return train_set , test_set
        
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        
    
    # This method initiates data ingestion
    
    def  initiate_data_ingestion(self) -> Data_Ingestion_Artifacts :
        """
        Method Name : initiate_data_ingestion
        
        Description : This method initiates data ingestion
        
        Output : Data ingestion artifacts
        """
        
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        
        try : 
            # Getting data from MongoDB
            df = self.get_data_from_mongodb()
            
            #Dropping the unneccessary columns from dataframe
            df1 = df.drop(self.data_ingestion_config.DROP_COLS,axis=1)
            df1 = df1.dropna()
            logging.info("Got the data from mongodb")
            
            
            # Splitting the data as train set and test set
            self.split_data_as_train_test(df1)
            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
            
            # Saving data ingestion artifacts
            data_ingestion_artifacts = Data_Ingestion_Artifacts(
                train_data_file_path = self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                test_data_file_path = self.data_ingestion_config.TEST_DATA_FILE_PATH
                )
            
            return data_ingestion_artifacts
        
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
            
        
    

