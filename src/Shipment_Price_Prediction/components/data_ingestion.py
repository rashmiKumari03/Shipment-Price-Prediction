
import sys
import os

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.impute import SimpleImputer

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

   b) Handle_Duplicates:
      - Input:
        - Reads the raw dataset from the feature store.
        - Identifies and removes duplicate entries based on predefined criteria.
      - Output:
        - Saves the cleaned dataset (with duplicates removed) back to the feature store.

   c) Handle_Missing_Values:
      - Input:
        - Reads the dataset with duplicates removed.
        - Identifies and handles missing values (e.g., by imputation, removal, or flagging).
      - Output:
        - Saves the dataset with handled missing values back to the feature store.

   d) Create_Target_Variable:
      - Input:
        - Reads the cleaned dataset with handled missing values.
        - Creates the target variable (e.g., "Shipment Price") based on business logic or data analysis.
      - Output:
        - Adds the target variable to the dataset and saves the updated dataset to the feature store.

   e) Drop_Columns:
      - Input:
        - Reads the feature store dataset with the target variable.
        - Uses a schema file to identify columns to drop based on business requirements.
      - Output:
        - Saves the final cleaned dataset (with dropped columns) back to the feature store.

   f) Split_Data_as_train_and_test:
      - Input:
        - Reads the cleaned and final dataset from the feature store.
        - Splits the data into train and test sets based on train_test_split_ratio.
      - Output:
        - Stores the train_data and test_data as train.csv and test.csv under Data_Ingestion_Artifact.

4. Final Outputs:
   - Feature_Store: Contains the complete cleaned dataset with target variable.
   - Train_Test_Split: Train and test datasets saved as:
     - Data_Ingestion_Artifact/Train/train.csv
     - Data_Ingestion_Artifact/Test/test.csv

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
    │      Handle_Duplicates       │
    │  - Remove duplicates         │
    │  - Save cleaned dataset      │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │    Handle_Missing_Values     │
    │  - Impute or remove missing  │
    │    values                    │
    │  - Save cleaned dataset      │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │     Create_Target_Variable   │
    │  - Create the target column  │
    │  - Save updated dataset      │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │         Drop_Columns         │
    │  - Load schema file          │
    │  - Drop unwanted columns     │
    │  - Save final cleaned dataset│
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

Purpose:
This workflow ensures modular, reproducible, and efficient data ingestion for machine learning pipelines, with clear separation of raw data, cleaned data, and training/testing datasets. The flow allows for seamless handling of duplicates, missing values, target variable creation, and data splitting to ensure data quality and readiness for modeling.
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
            logging.info("Data fetched successfully from MongoDB.")
            logging.info("Exited the get_data_from_mongodb method of Data_Ingestion class")
            
            return df
        
        except Exception as e :
            raise CustomException(f"Error in fetching data from MongoDB: {str(e)}", sys)
          
          
          
      
    # Preprocess a bit and creating the Target column    
    def handle_duplicate_values(self, data: DataFrame) -> DataFrame:
        """
        Method Name : handle_duplicates
        
        Description : This method handles duplicate rows in the dataset by checking for exact duplicates and removing them.
                      It ensures that only unique rows are retained in the dataset.
        
        Output      : DataFrame with duplicates removed.
        """
        logging.info("Entered handle_duplicates method of Data_Ingestion class")
        
        try:
            # Check for duplicate rows
            duplicates_count = data.duplicated().sum()

            if duplicates_count > 0:
                logging.info(f"Found {duplicates_count} duplicate rows. Removing duplicates.")
                data = data.drop_duplicates()
                logging.info(f"Duplicates removed. The new number of rows: {data.shape[0]}")
            else:
                logging.info("No duplicates found in the dataset.")
            
            logging.info("Exited handle_duplicates method of Data_Ingestion class")
            return data

        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

          
    
          

    def handle_missing_values(self, data: DataFrame) -> DataFrame:
        """
        Method Name : handle_missing_values
        
        Description : This method handles missing values in the dataset by applying imputation strategies. 
                      Numerical columns are imputed with the median value, categorical columns are imputed 
                      with the most frequent value, and datetime columns are processed to extract meaningful features.
                      No imputation is done for datetime columns as there are no missing values.
        
        Output      : DataFrame with missing values imputed and date features transformed into meaningful columns.
        """
        logging.info("Entered handle_missing_values method of Data_Ingestion class")
        try:
            # Get columns from SCHEMA_CONFIG
            ele_target_cols = self.data_ingestion_config.SCHEMA_CONFIG["creating_target_columns"]
            
            # Impute missing values for only the column we need to create target column
            # Convert non-numeric values to NaN in numerical columns
            data[ele_target_cols] = data[ele_target_cols].apply(pd.to_numeric, errors='coerce')

            numerical_imputer = SimpleImputer(strategy='median')
            data[ele_target_cols] = numerical_imputer.fit_transform(data[ele_target_cols])

            logging.info("Exited handle_missing_values method of Data_Ingestion class")
            return data

        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

        
        

    def create_target_column(self, data: DataFrame) -> DataFrame:
      """
      Method Name : create_target_column
      Description : This method constructs the target column 'Shipment Price' by summing up the values of
                    specific columns listed in the 'creating_target_columns' configuration.
                    It handles the creation of the 'Shipment Price' in the dataframe.
      
      Output      : Updated dataframe with the new 'Shipment Price' target column.
      """
      logging.info("Entered create_target_column method of Data_Ingestion class")
      
      try:
          # Fetch the columns specified in the configuration
          ele_target_cols = self.data_ingestion_config.SCHEMA_CONFIG["creating_target_columns"]

          # Ensure the columns exist in the DataFrame before processing
          missing_columns = [col for col in ele_target_cols if col not in data.columns]
          if missing_columns:
              raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")

          # Sum the relevant columns to create the 'Shipment Price' target column
          data["Shipment Price"] = data[ele_target_cols].sum(axis=1)

          logging.info("Exited create_target_column method of Data_Ingestion class")
          return data

      except Exception as e:
          logging.info(CustomException(str(e), sys))
          raise CustomException(str(e), sys)


        
       
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
            train_set , test_set =  train_test_split(df , test_size = TEST_SIZE )
            logging.info("Performed train test split on the dataframe")
            
            # Creating train directory under data ingestion artifact directory
            os.makedirs(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_FILE_DIR,exist_ok=True)
            logging.info(f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR)} directory")
            
            # Creating test directory under data ingestion artifact directory
            os.makedirs(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR,exist_ok=True)
            logging.info(f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTIFACT_FILE_DIR)} directory")
            
            
            # Saving train.csv file to train directory
            train_set.to_csv(self.data_ingestion_config.TRAIN_DATA_FILE_PATH,index = False,header = True)
            
            # Saving test.csv file to test directory
            test_set.to_csv(self.data_ingestion_config.TEST_DATA_FILE_PATH,index = False,header = True)
            
            logging.info("Converted Train DataFrame and Test DataFrame into csv")
            logging.info(f"Saved {os.path.basename(self.data_ingestion_config.TRAIN_DATA_FILE_PATH)},{os.path.basename(self.data_ingestion_config.TEST_DATA_FILE_PATH)} in {os.path.basename(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)}.")
            
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            
            return train_set , test_set
  
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    
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
            raw_data = self.get_data_from_mongodb()
            
            #Dropping the unneccessary columns 
            cleaned_data = raw_data.drop(columns=self.data_ingestion_config.DROP_COLS , errors='ignore')
            logging.info("After Dropping the columns we have the dataset as")
            logging.info(cleaned_data.head())
            logging.info("Got the data from mongodb")
            
            
            # Step 2: Handle Duplicates and Missing Values for cleaned_data

            cleaned_data = self.handle_duplicate_values(cleaned_data)
            logging.info("Handled duplicate values in cleaned dataset.")
            logging.info(f"Duplicate values in cleaned_data now: {cleaned_data.duplicated().sum()}")

            cleaned_data = self.handle_missing_values(cleaned_data)
            logging.info("Handled missing values in cleaned dataset.")
            logging.info(f"Cleaned dataset after missing value imputation:\n {cleaned_data.head()}")

            # Step 3: Create the target column 'Shipment Price' in cleaned_data
            logging.info("Creating target column 'Shipment Price' in cleaned dataset...")
            cleaned_data = self.create_target_column(cleaned_data)
            logging.info(f"Cleaned dataset with 'Shipment Price' column:\n {cleaned_data.head()}")

            
            
            # Splitting the data as train set and test set
            self.split_data_as_train_test(cleaned_data)
            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
            
            # Saving data ingestion artifacts
            data_ingestion_artifacts = Data_Ingestion_Artifacts(
                train_data_file_path = self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                test_data_file_path = self.data_ingestion_config.TEST_DATA_FILE_PATH
                )
            logging.info("Data Ingestion process completed successfully.")
            return data_ingestion_artifacts
        
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
            
        
    

