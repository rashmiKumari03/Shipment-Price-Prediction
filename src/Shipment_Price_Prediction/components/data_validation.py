import sys
import os
import json
import pandas as pd
import pandas as DataFrame

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from typing import Tuple,Union
from evidently.model_profile import Profile 
from evidently.model_profile.sections import DataDriftProfileSection

from src.Shipment_Price_Prediction.entity.config_entity import Data_Validation_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import(Data_Ingestion_Artifacts,Data_Validation_Artifacts)

# We need to mention some of the constant to the constant folder.
# Creating Data_Validation_Config in config_entity  also Data_Validation_Artifacts in artifacts_entity and call them here.

class Data_Validation:
    def __init__(self,
                 data_ingestion_artifacts:Data_Ingestion_Artifacts,
                 data_validation_config: Data_Validation_Config
                 ):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_validation_config = data_validation_config
        
    # This method is used to validate schema columns
    def validate_schema_columns(self,df:DataFrame) -> bool :
        """
        Method Name : validate_schema_columns
        
        Description : This method validates schema columns of dataframe.
        
        Output      : True or False
        
        """
        
        try:
            # Checking the len of Dataframe columns and Schema file columns
            if len(df.columns) == len(self.data_validation_config.SCHEMA_CONFIG["columns"]):
                validation_status = True
            else:
                validation_status = False
            return validation_status
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
    def is_numerical_column_exists(self,df: DataFrame) -> bool:
        """
        Method Name : is_numerical_column_exists
        
        Description : This method validates wheather a numerical column exists in the dataframe.
        
        Output      : True or False
        
        """
        try:
            validation_status = False
            
            # Checking numerical schema columns with dataframe numerical columns
            for column in self.data_validation_config.SCHEMA_CONFIG["numerical_columns"]:
                if column not in df.columns:
                    logging.info(f"Numerical Column - {column} not found in the DataFrame")
                else:
                    validation_status = True
                
                return validation_status
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
        
    def is_categorical_column_exists(self,df:DataFrame) -> bool :
        """
        Method Name : is_categorical_column_exists
        
        Description : This method validates wheather a categorical column exists in the dataframe.
        
        Output : True or False
        
        """
        try:
            validation_status = False
            
            # Checking categorical schema columns with data frame categorical columns
            for column in self.data_validation_config.SCHEMA_CONFIG["categorical_columns"]:
                if column not in df.columns:
                    logging.info(f"Categorical Column - {column} not found in DataFrame")
                else:
                    validation_status = True
                return validation_status
             
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
        
    def validate_dataset_schema_columns(self) -> Tuple[bool,bool]:
        """
        Method Name : validate_dataet_schema_columns
        
        Description : This method validates schema for Train DataFrame and Test DataFrame.
        
        Outpu       : True or False
        
        """
        logging.info("Entered validate_dataset_schema_columns method of Data_Validation class")
        
        try:
            logging.info("Validation dataset schema columns")
            
            # Validating Schema columns for Train DataFrame
            train_schema_status = self.validate_schema_columns(self.train_set)
            logging.info("Validated dataset schema columns on the train set")
            
            # Validating Schema columns for Test DataFrame
            test_schema_status = self.validate_schema_columns(self.test_set)
            logging.info("Validated dataset schema columns on the test set")
            
            return train_schema_status , test_schema_status
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
        
    def validate_is_numerical_column_exists(self) -> Tuple[bool,bool]: 
        """
        Method Name : validate_is_numerical_column_exists
        
        Description : This method validates wheather numerical columns exists for Train DataFrame and Test DataFrame. 
        
        Output      : True or False 
        
        """
        logging.info("Entered validate_is_numerical_column_exists method of Data_Validation class")
        try :
            logging.info("Validating dataset schema for numericl datatypes.")
            
            # Validating numerical columns with Train DataFrame
            train_num_datatype_status = self.is_numerical_column_exists(self.train_set)
            logging.info("Validated dataset schema for numerical datatypes for trainset.")
            
            # Validating numerical columns with Test DataFrame.
            test_num_datatype_status = self.is_numerical_column_exists(self.test_set)
            logging.info("Validated dataset schema for numerical datatype for testset.")
            
            logging.info("Exited validate_is_numerical_column_exists method of Data_Validation class.")
            
            
            return train_num_datatype_status , test_num_datatype_status
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
    
    def validate_is_categorical_column_exists(self) -> Tuple[bool,bool]:
        """
        Method Name : validate_is_categorical_column_exists
        
        Description : This method validates wheather categorical columns exists for Train Dataframe and Test DataFrame. 
        
        Output      : True or False
        
        """
        logging.info("Entered validate_is_categorical_columns_exists method of Data_Validation class.")
        try:
            logging.info("Validating dataset schema for categorical datatype")
            
            # Validating categorical columns with Train DataFrame
            train_cat_datatype_status = self.is_categorical_column_exists(self.train_set)
            logging.info("Validated dataset schema for categorical datatype for trainset.")
            
            # Validating categorical columns with Test DataFrame
            test_cat_datatype_status = self.is_categorical_column_exists(self.test_set)
            logging.info("Validated dataset schema for categorical datatype for testset.")
            
            logging.info("Exited validate_is_categorical_column_exists method of Data_Validation class.")
            
            return train_cat_datatype_status , test_cat_datatype_status
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
        
    def detect_dataset_drift(self, reference: DataFrame, production: DataFrame, get_ratio: bool = False) -> Union[bool, float]:
        """
        Method Name : detect_dataset_drift
        
        Description : This method detects whether data drift is present in the dataset by comparing the reference dataset (train) 
                        with the production dataset (test). Data drift occurs when the statistical properties of a dataset change 
                        between different periods or environments, which can impact model performance.
        
        Input       : reference (DataFrame) - Reference dataset (typically the training dataset)
                        production (DataFrame) - Production dataset (typically the testing dataset)
                        get_ratio (bool) - Flag to decide whether to return drift ratio (True) or drift status (False)
                        
        Output      : If get_ratio is True, returns the drift ratio (float), 
                        otherwise returns drift status (True if drift is present, False otherwise).
        """
        try:
            # Initializing the data drift profile to analyze drift between the two datasets
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference, production)
            
            # Generating the data drift report in JSON format
            report = data_drift_profile.json()
            json_report = json.loads(report)
            
            # Saving the JSON report to the artifacts directory for further analysis
            data_drift_file_path = self.data_validation_config.DATA_DRIFT_FILE_PATH
            self.data_validation_config.UTILS.write_json_to_yaml_file(json_report, data_drift_file_path)
            
            # Extracting the number of features and the number of drifted features from the JSON report
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted"]
            
            if get_ratio:
                # Returning the ratio of drifted features to the total number of features
                return n_drifted_features / n_features
            else:
                # Returning whether dataset drift is detected (True/False)
                return json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
        
        except Exception as e:
            # Logging and raising custom exceptions for better error tracking
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        
        
        

    def initiate_data_validation(self) -> Data_Validation_Artifacts:
        """
        Method Name : initiate_data_validation
        
        Description : This method initiates the data validation process by validating schema, numerical columns, 
                    categorical columns, and detecting dataset drift.
        
        Output      : Returns a Data_Validation_Artifacts object containing the data drift file path 
                    and the overall validation status.
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")
        try:
            # Reading the train and test datasets from data ingestion artifacts directory
            self.train_set = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
            self.test_set = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)
            
            logging.info("Initiated data validation for the dataset")
            
            # Creating the directory to save data validation artifacts
            os.makedirs(self.data_validation_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.data_validation_config.DATA_DRIFT_FILE_PATH)}")
            
            # Checking for dataset drift between the train and test datasets
            drift = self.detect_dataset_drift(self.train_set, self.test_set)
            
            # Validating schema columns for train and test datasets
            schema_train_cols_status, schema_test_cols_status = self.validate_dataset_schema_columns()
            logging.info(f"Schema train columns status is {schema_train_cols_status} and schema test columns status is {schema_test_cols_status}")
            
            # Validating that numerical columns are present in train and test datasets
            schema_train_num_cols_status, schema_test_num_cols_status = self.validate_is_numerical_column_exists()
            logging.info(f"Schema train numerical columns status is {schema_train_num_cols_status} and schema test columns status is {schema_test_num_cols_status}")
            
            # Validating that categorical columns are present in train and test datasets
            schema_train_cat_cols_status, schema_test_cat_cols_status = self.validate_is_categorical_column_exists()
            logging.info(f"Schema train categorical columns status is {schema_train_cat_cols_status} and schema test categorical columns status is {schema_test_cat_cols_status}")
            
            # Combining all validation results to determine overall drift status
            drift_status = None  # Initial drift status is None
            if (
                schema_train_cols_status is True and
                schema_test_cols_status is True and
                schema_train_num_cols_status is True and
                schema_test_num_cols_status is True and
                schema_train_cat_cols_status is True and
                schema_test_cat_cols_status is True and
                drift is False
            ):
                logging.info("Dataset schema validation completed successfully")
                drift_status = True  # All checks passed
            else:
                drift_status = False  # One or more checks failed
            
            # Saving data validation artifacts with the drift status and report path
            data_validation_artifacts = Data_Validation_Artifacts(
                data_drift_file_path= self.data_validation_config.DATA_DRIFT_FILE_PATH,
                validation_status=drift_status
            )
            return data_validation_artifacts

        except Exception as e:
            # Logging and raising custom exceptions for error tracking
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        
# After this Lets go to Training Pipeline and code ...