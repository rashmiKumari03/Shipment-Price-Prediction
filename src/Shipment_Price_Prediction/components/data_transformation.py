# Importing necessary libraries
import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import skew
from scipy.stats.mstats import winsorize

# Custom imports for logging and exception handling
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.entity.config_entity import Data_Transformation_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts, Data_Validation_Artifacts, Data_Transformation_Artifacts)

# Creating Data_Transformation class
class Data_Transformation:
    """
    This class handles data transformation tasks such as data type conversion, missing value imputation, 
    outlier detection, and preprocessing. It applies necessary transformations to the dataset and returns 
    processed artifacts for downstream tasks.
    """

    def __init__(self, 
                 data_ingestion_artifacts: Data_Ingestion_Artifacts, 
                 data_transformation_config: Data_Transformation_Config
                 ):
        """
        Initializes the Data_Transformation class by accepting ingestion artifacts and transformation configurations.
        
        Args:
            data_ingestion_artifacts (Data_Ingestion_Artifacts): The artifacts produced by data ingestion.
            data_transformation_config (Data_Transformation_Config): Configuration for the data transformation process.
        """
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config

        # Reading the train.csv and test.csv from the data ingestion artifacts
        self.train_set = pd.read_csv(self.data_ingestion_artifacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artifacts.test_data_file_path)

    def segregate_datatype(self, data: DataFrame) -> DataFrame:
        """
        Method Name : handle_data_types
        
        Description : This method handles the segregate the data types for columns. It ensures that numerical 
                      columns are of numeric type, categorical columns are converted to categorical type, and 
                      datetime columns are appropriately parsed.

        Output      : DataFrame with columns converted to the appropriate data types.
        """
        logging.info("Entered handle_data_types method of Data_Transformation class")
        try:
            # Converting numerical columns
            numerical_cols = data.select_dtypes(include=['number']).columns
            data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')

            # Converting categorical columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                data[col] = data[col].astype("category")

            # Converting datetime columns
            datetime_cols = data.select_dtypes(include=['datetime']).columns
            for col in datetime_cols:
                data[col] = pd.to_datetime(data[col], errors='coerce')

            logging.info("Data types have been successfully converted")
            logging.info("Exited handle_data_types method of Data_Transformation class")
            return data
        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        
        
        
     # There must be the datatype conversion from any dataype to required required ( might be using schema.yaml it will work.)

    def handle_missing_values(self, data: DataFrame) -> DataFrame:
        """
        Method Name : handle_missing_values
        
        Description : This method handles missing values in the dataset by applying imputation strategies. 
                      Numerical columns are imputed with the median value, categorical columns are imputed 
                      with the most frequent value, and datetime columns are imputed with the most frequent value.

        Output      : DataFrame with missing values imputed.
        """
        logging.info("Entered handle_missing_values method of Data_Transformation class")
        try:
            # Separate columns into numerical, categorical, and datetime
            numerical_cols = data.select_dtypes(include=['number']).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            datetime_cols = data.select_dtypes(include=['datetime']).columns

            # Imputing numerical columns using median
            numerical_imputer = SimpleImputer(strategy='median')
            data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
            logging.info(f"Imputed missing values in numerical columns: {numerical_cols}")
            
            # Imputing categorical columns using most frequent value
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
            logging.info(f"Imputed missing values in categorical columns: {categorical_cols}")
            
            # Imputing datetime columns using most frequent value (if datetime column exists)
            if len(datetime_cols) > 0:
                datetime_imputer = SimpleImputer(strategy='most_frequent')
                data[datetime_cols] = datetime_imputer.fit_transform(data[datetime_cols])
                logging.info(f"Imputed missing values in datetime columns: {datetime_cols}")
            
            logging.info("Exited handle_missing_values method of Data_Transformation class")
            return data
        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def handle_outliers(self, data: DataFrame) -> DataFrame:
        """
        Method Name : handle_outliers
        
        Description : This method handles outliers in both numerical and categorical columns using 
                      Winsorization to limit extreme values based on the column's skewness.

        Output      : DataFrame with outliers handled by capping extreme values.
        """
        logging.info("Entered handle_outliers method of Data_Transformation class")
        try:
            # Handle outliers in numerical columns using Winsorization based on skewness
            numerical_cols = data.select_dtypes(include=['number']).columns
            for col in numerical_cols:
                # Calculate skewness
                skewness = skew(data[col].dropna())
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
                data[col] = winsorize(data[col], limits=[lower_limit, upper_limit])
                logging.info(f"Outliers capped for numerical column: {col}")

            logging.info("Exited handle_outliers method of Data_Transformation class")
            return data
        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def get_data_transformer_object(self) -> object:
        """
        Method Name : get_data_transformer_object
        
        Description : This method returns a preprocessor object that applies various data transformations such as 
                      imputation, encoding, and scaling to the dataset. The method builds a transformation pipeline 
                      using SimpleImputer, OneHotEncoder, and BinaryEncoder to process different feature types.

        Output      : Preprocessor object (ColumnTransformer) for transforming the dataset.
        """
        logging.info("Entered get_data_transformer_object method of Data_Transformation class")
        try:
            # Getting necessary column names from config file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            one_hot_columns = self.data_transformation_config.SCHEMA_CONFIG["one_hot_columns"]
            binary_columns = self.data_transformation_config.SCHEMA_CONFIG["binary_columns"]

            # Create transformer objects for missing value imputation, encoding, and scaling
            numeric_imputer = SimpleImputer(strategy="mean")  # Impute missing values for numerical columns
            categorical_imputer = SimpleImputer(strategy="most_frequent")  # Impute missing values for categorical columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', numeric_imputer),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', categorical_imputer),
                ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore"))
            ])

            binary_transformer = Pipeline(steps=[
                ('binary_encoder', BinaryEncoder())
            ])

            logging.info("Initialized Imputers, StandardScaler, OneHotEncoder, BinaryEncoder")

            # Using transformer objects in column transforms:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("one_hot", categorical_transformer, one_hot_columns),
                    ("binary", binary_transformer, binary_columns),
                    ("numeric", numeric_transformer, numerical_columns)
                ])

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of Data_Transformation class")

            return preprocessor

        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def apply_transformations(self) -> Data_Transformation_Artifacts:
        """
        Method Name : apply_transformations
        
        Description : This method applies the entire data transformation process to the train and test datasets. 
                      It handles data type conversion, missing value imputation, outlier handling, and feature 
                      encoding and scaling. The transformed data is then saved as artifacts for further processing.

        Output      : Data_Transformation_Artifacts containing processed data and model-ready artifacts.
        """
        logging.info("Entered apply_transformations method of Data_Transformation class")
        try:
            # Creating directory for data transformation artifacts
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Created artifacts directory for {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)}")
            
            # Step 1: Data Type Conversion
            self.train_set = self.handle_data_types(self.train_set)
            self.test_set = self.handle_data_types(self.test_set)
            logging.info("Data types converted for train and test datasets")

            # Step 2: Handle Missing Values
            self.train_set = self.handle_missing_values(self.train_set)
            self.test_set = self.handle_missing_values(self.test_set)
            logging.info("Missing values handled for train and test datasets")

            # Step 3: Handle Outliers
            self.train_set = self.handle_outliers(self.train_set)
            self.test_set = self.handle_outliers(self.test_set)
            logging.info("Outliers handled for train and test datasets")

            # Step 4: Apply Feature Transformation Pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training data
            X_train_transformed = preprocessor.fit_transform(self.train_set)
            logging.info("Transformed training data using preprocessor")

            # Transform the test data
            X_test_transformed = preprocessor.transform(self.test_set)
            logging.info("Transformed test data using preprocessor")

            # Saving processed data as artifacts
            transformed_train_file_path = os.path.join(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, "transformed_train_data.csv")
            transformed_test_file_path = os.path.join(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, "transformed_test_data.csv")

            # Saving the transformed data to CSV
            pd.DataFrame(X_train_transformed).to_csv(transformed_train_file_path, index=False)
            pd.DataFrame(X_test_transformed).to_csv(transformed_test_file_path, index=False)
            logging.info("Transformed data saved as CSV")

            # Creating and returning Data_Transformation_Artifacts
            data_transformation_artifacts = Data_Transformation_Artifacts(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path
            )
            
            logging.info("Exited apply_transformations method of Data_Transformation class")
            return data_transformation_artifacts

        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
