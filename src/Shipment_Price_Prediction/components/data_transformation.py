import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from category_encoders.binary import BinaryEncoder

from scipy.stats import zscore, skew
from scipy.stats.mstats import winsorize

from src.Shipment_Price_Prediction.entity.config_entity import Data_Transformation_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts,Data_Validation_Artifacts,Data_Transformation_Artifacts)

# Initialize the constant 
# Then go to config_entity and artifacts_entity to code for data validation.

# Creating Data_Transformation class

class Data_Transformation:
    def __init__(self, data_ingestion_artifacts, data_transformation_config):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config

        # Reading the train.csv and test.csv from the data ingestion artifacts
        self.train_set = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)

    def get_data_transformer_object(self) -> object:
        """
        Method Name : get_data_transformer_object

        Description : This method gives preprocessor objects.

        Output      : Preprocessor object
        """
        logging.info("Entered get_data_transformer_object method of Data_Transformation class")
        try:
            # Getting necessary column names from config file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            one_hot_columns = self.data_transformation_config.SCHEMA_CONFIG["one_hot_columns"]
            binary_columns = self.data_transformation_config.SCHEMA_CONFIG["binary_columns"]

            # Creating transformer objects
            numeric_transformer = StandardScaler()
            one_hot_transformer = OneHotEncoder(handle_unknown="ignore")
            binary_transformer = BinaryEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, BinaryEncoder")

            # Using transformer objects in column transforms:
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", one_hot_transformer, one_hot_columns),
                    ("BinaryEncoder", binary_transformer, binary_columns),
                    ("StandardScaler", numeric_transformer, numerical_columns),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info("Exited get_data_transformer_object method of Data_Transformation class")

            return preprocessor

        except Exception as e:
            raise CustomException(str(e), sys)

     
    @staticmethod
    def outlier_handler(col: str, df: DataFrame) -> DataFrame:
        """
        Method Name : outlier_handler

        Description : This method handles outliers using Winsorization based on skewness.

        Output      : DataFrame with outliers capped for the specified column.
        """
        try:
            # Calculate skewness
            skewness = skew(df[col].dropna())
            logging.info(f"Skewness of {col}: {skewness}")

            # Define Winsorization limits based on skewness
            if skewness < -5:
                lower_limit, upper_limit = 0.25, 0.02
            elif -5 <= skewness < -2:
                lower_limit, upper_limit = 0.23, 0.02
            elif -2 <= skewness < -0.1:
                lower_limit, upper_limit = 0.05, 0.02
            elif -0.1 <= skewness <= 0.1:
                lower_limit, upper_limit = 0.001, 0.001
            elif 0.1 < skewness <= 2:
                lower_limit, upper_limit = 0.02, 0.05
            elif 2 < skewness <= 5:
                lower_limit, upper_limit = 0.02, 0.23
            elif skewness > 5:
                lower_limit, upper_limit = 0.02, 0.25

            logging.info(f"Applying Winsorization with limits: Lower = {lower_limit}, Upper = {upper_limit}")

            # Apply Winsorization
            df[col] = winsorize(df[col], limits=[lower_limit, upper_limit])

            logging.info(f"Outliers capped for column: {col}")

            return df

        except Exception as e:
            raise CustomException(str(e), sys)
        
        
    # This method is used to initialize data transformation.
    def initiate_data_transformation(self) -> Data_Transformation_Artifacts:
        """ 
        Method Name : initiate_data_transformation
        
        Description : This method initiates data transformation
        
        Output      : Data Transformation Artifacts
        """
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")
        try:
            
            # Creating directory for data transformation artifacts
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,exist_ok= True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Getting preprocessor object
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the Preprocessor object")
            
            # Getting target column name from schema file
            target_column_name = self.data_transformation_config.SCHEMA_CONFIG["target_column"]
            
            # Getting numerical columns from schema file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            logging.info("Got target column name and numerical columns from schema config")
            
            continuous_columns = [
                feature
                for feature in numerical_columns 
                if len(self.train_set[feature].unique() >= 25)]
            
            logging.info("Got a list of continuous_columns")
            [self.outlier_handler(col,self.train_set) for col in continuous_columns]
            logging.info("Outliers Handled in train dataset")
            
            [self.outlier_handler(col,self.test_set) for col in continuous_columns]
            logging.info("Outlier Handled in test dataset")
            
            
            # Getting the input features and target features of Training dataset
            input_features_train_df = self.train_set.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = self.train_set[target_column_name]
            
             # Getting the input features and target features of Testing dataset
            input_features_test_df = self.test_set.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = self.test_set[target_column_name]
            
            logging.info("Got train features and test features")
            
            
            
            # Applying preprocessing object on training dataframe and testing dataframe
            # On Train do : fit and transform and On Test do : transform
            input_feature_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor.transform(input_features_test_df)
            logging.info("Used the preprocessor object to transform the test features")
            
            
            # Concatenating input feature array and target feature array of Train Dataset.
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            logging.info("Created Train array")
            
            # Creating the directory for transformed traind dataset array and saving the array.
            os.makedirs(self.data_transformation_config.TRANSFORMED_TRAIN_DATA_DIR, exist_ok= True)
            
            transformed_train_file = self.data_transformation_config.UTILS.save_numpy_array_data(self.data_transformation_config.TRANSFORMED_TRAIN_FILE_PATH,train_arr)
            logging.info(f"Saved train array to {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS)}")
            
            # Concatenating input feature array and target feature array of Test dataset
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]
            logging.info("Created Test array")
            
            # Creating directory for transformed test dataset array and saving the array
            os.makedirs(self.data_transformation_config.TRANSFORMED_TEST_DATA_DIR,exist_ok= True)
            
            transformed_test_file = self.data_transformation_config.UTILS.save_numpy_array_data(self.data_transformation_config.TRANSFORMED_TEST_FILE_PATH,test_arr)
            logging.info(f"Saved test array to {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS)}")
            
            # Saving the preprocessor object to data transformation artifacts directory.
            preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(self.data_transformation_config.PREPROCESSOR_FILE_PATH,preprocessor)
            logging.info("Saved the preprocessor object in Data Transformation Artifacts Directory")
            
            logging.info("Exited initiate_data_transformation method of Data_Transformation class")
            
            
            # Saving the data transformation artifacts:
            data_transformation_artifacts = Data_Transformation_Artifacts(
                transformed_object_file_path= preprocessor_obj_file,
                transformed_train_file_path= transformed_train_file,
                transformed_test_file = transformed_test_file
            )
            
            return data_transformation_artifacts
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
            
    

                
# To run this we need to call this class in training_pipeline..
            
            
         