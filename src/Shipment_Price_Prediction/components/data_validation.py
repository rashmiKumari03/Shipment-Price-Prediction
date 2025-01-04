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
        
        Description : This method validates wheather numerical columns exists for Train DataFrame and 
        
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
        
                         

