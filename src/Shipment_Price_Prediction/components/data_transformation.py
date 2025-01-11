# Importing necessary libraries
import os
import sys
import yaml
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import skew
from scipy.stats.mstats import winsorize
import joblib

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
                
    """
    Note:
    We do not need to explicitly typecast the data types of the columns in this function, as the data type conversion has already been handled in the schema.yaml file. 
    In the schema.yaml, we define the expected data types for each column before the data is processed. This ensures that when the dataset is loaded or read into the system, 
    it automatically adheres to the specified data types as per the schema. Therefore, there is no need to manually typecast the columns again in this part of the code.

    """


    def basic_inspection(self):
        """
        Method Name : basic_inspection

        Description : This method performs an initial inspection of the dataset by logging detailed information.
                    It provides insights into the dataset's shape, data types of each column, missing values,
                    duplicate rows, and basic descriptive statistics such as mean, median, standard deviation, etc.
                    The method helps to quickly assess the quality and structure of the data.

        Output      : Logs detailed dataset information, including:
                    - Shape of the dataset (number of rows and columns)
                    - Data types of each column
                    - Count of missing values in each column
                    - Number of duplicate rows
                    - Descriptive statistics (mean, std, min, 25%, 50%, 75%, max) for numerical columns
        """
        logging.info("Entered basic_inspection method of Data_Transformation class")
        try:
            data = {"Training_dataset": self.train_set, "Testing_dataset": self.test_set}
            
            for dataset_name, dataset in data.items():
                logging.info(f"Inspection of {dataset_name}:")
                logging.info(f"The Number of Rows/Records in the dataset: {dataset.shape[0]}")
                logging.info(f"The Number of Columns/Features in the dataset: {dataset.shape[1]}")
                logging.info(f"Dataset Preview (First 5 records):\n{dataset.head()}")

                # Log data types and general info about the dataset
                logging.info("Data Information:")
                logging.info(dataset.info())

                # Log missing values count in each column
                logging.info("Checking for Missing (Null) Values in each column:")
                logging.info(dataset.isnull().sum())

                # Log duplicate rows count
                logging.info("Checking for Duplicates in the dataset:")
                logging.info(dataset.duplicated().sum())

                # Log descriptive statistics for numerical and categorical columns
                logging.info("Descriptive Statistics of Numerical and Categorical Data:")
                logging.info("Numerical Data Statistics:")
                logging.info(dataset.describe(include=['number']))
                
                logging.info("Categorical Data Statistics:")
                logging.info(dataset.describe(include=['object']))

                # Segregate columns by type using SCHEMA_CONFIG
                numerical_cols = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
                categorical_cols = self.data_transformation_config.SCHEMA_CONFIG["categorical_columns"]
                datetime_cols = self.data_transformation_config.SCHEMA_CONFIG["datetime_columns"]

               
                logging.info(f"Converted {len(numerical_cols)} numerical columns to numeric type.Those are : {numerical_cols}")
                logging.info(f"Converted {len(categorical_cols)} categorical columns to string type.Those are :{categorical_cols}")
                logging.info(f"Converted {len(datetime_cols)} datetime columns to datetime type.Those are :{datetime_cols}")

            logging.info("Exited basic_inspection method of Data_Transformation class")
            return None

        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        

    
    def handle_missing_values(self, data: DataFrame) -> DataFrame:
        """
        Method Name : handle_missing_values
        
        Description : This method handles missing values in the dataset by applying imputation strategies. 
                      Numerical columns are imputed with the median value, categorical columns are imputed 
                      with the most frequent value, and datetime columns are processed to extract meaningful features.
                      No imputation is done for datetime columns as there are no missing values.
        
        Output      : DataFrame with missing values imputed and date features transformed into meaningful columns.
        """
        logging.info("Entered handle_missing_values method of Data_Transformation class")
        try:
            # Get columns from SCHEMA_CONFIG
            numerical_cols = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            categorical_cols = self.data_transformation_config.SCHEMA_CONFIG["categorical_columns"]
            
            # Impute missing values for numerical columns
            if numerical_cols:
                # Convert non-numeric values to NaN in numerical columns
                data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')

                numerical_imputer = SimpleImputer(strategy='median')
                data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])

            # Impute missing values for categorical columns
            if categorical_cols:

                categorical_imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

            logging.info("Exited handle_missing_values method of Data_Transformation class")
            return data
    
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
        
        
        
    def handle_duplicate_values(self,data:DataFrame) -> DataFrame:
        """
        Method Name : handle_duplicates
        
        Description : This method handles duplicate rows in the dataset by checking for exact duplicates and removing them.
                    It ensures that only unique rows are retained in the dataset.
        
        Output      : DataFrame with duplicates removed.
        """
        logging.info("Entered handle_duplicates method of Data_Transformation class")
        
        try:
            # Check for duplicate rows
            duplicates_count = data.duplicated().sum()

            if duplicates_count > 0:
                logging.info(f"Found {duplicates_count} duplicate rows. Removing duplicates.")
                data = data.drop_duplicates()
                logging.info(f"Duplicates removed. The new number of rows: {data.shape[0]}")
            else:
                logging.info("No duplicates found in the dataset.")
            
            logging.info("Exited handle_duplicates method of Data_Transformation class")
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
            
            # Get numerical columns from SCHEMA_CONFIG
            numerical_cols = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
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
        
        This method returns a preprocessor object that applies various data transformations such as 
        imputation, encoding, and scaling to the dataset. It creates a preprocessor pipeline with 
        SimpleImputer, OneHotEncoder, BinaryEncoder, and StandardScaler for different feature types.

        Output :
        Preprocessor object (ColumnTransformer) for transforming the dataset.
        """
        logging.info("Entered get_data_transformer_object method of Data_Transformation class")

        try:
            # Get the column names from the configuration
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            categorical_columns = self.data_transformation_config.SCHEMA_CONFIG["categorical_columns"]
            binary_columns = self.data_transformation_config.SCHEMA_CONFIG["binary_columns"]
            datetime_columns = self.data_transformation_config.SCHEMA_CONFIG["datetime_columns"]

            # Handle numerical columns: Impute and scale

            numeric_imputer = SimpleImputer(strategy="median")
            numeric_pipeline = Pipeline(steps=[
                ('imputer', numeric_imputer),
                ('scaler', StandardScaler())  # Standardize numerical data
            ])
            logging.info("Numerical pipeline created.")

            # Handle categorical columns: Impute and one-hot encode
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            categorical_pipeline = Pipeline(steps=[
                ('imputer', categorical_imputer),
                ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore"))  # One-hot encode categorical data
            ])
            logging.info("Categorical pipeline created.")

            # Handle binary columns: Apply binary encoding
            binary_pipeline = Pipeline(steps=[
                ('binary_encoder', BinaryEncoder())
            ])
            logging.info("Binary pipeline created.")
            
            
             
            # Handle datetime columns: Extract features from datetime columns
            def datetime_transformer(X):
                for col in datetime_columns:
                    if col in X.columns:
                        if not pd.api.types.is_datetime64_any_dtype(X[col]):
                            X[col] = pd.to_datetime(X[col], errors='coerce')  # Ensure datetime type
                        X[f'{col}_year'] = X[col].dt.year
                        X[f'{col}_month'] = X[col].dt.month
                        X[f'{col}_day'] = X[col].dt.day
                    else:
                        logging.warning(f"Column {col} not found in DataFrame.")
                return X.drop(columns=datetime_columns, errors='ignore')

            # Create pipeline for datetime transformation
            datetime_pipeline = Pipeline(steps=[('datetime_features', FunctionTransformer(func=datetime_transformer, validate=False))])
            logging.info("Datetime pipeline created.")
                
            
            # Combine all transformations into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_pipeline, numerical_columns),
                    ('categorical', categorical_pipeline, categorical_columns),
                    ('binary', binary_pipeline, binary_columns),
                    ('datetime', datetime_pipeline, datetime_columns)
                ]
            )

            logging.info("Data transformer object created successfully.")
            logging.info("Exited get_data_transformer_object method of Data_Transformation class")
            return preprocessor

        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
   
            
    def initiate_data_transformation(self) -> Data_Transformation_Artifacts:
        """
        Method Name : initiate_data_transformation
        
        Description : This method applies the entire data transformation process to the train and test datasets. 
                    It handles data type conversion, missing value imputation, outlier handling, and feature 
                    encoding and scaling. The transformed data is then saved as artifacts for further processing.

        Output      : Data_Transformation_Artifacts containing processed data and model-ready artifacts.
        """
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")

        try:
            # Create directory for data transformation artifacts if not exists
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Created artifacts directory: {self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR}")

            # Step 1: Basic Inspection of Train and Test sets
            logging.info("Basic Inspection of Train dataset and Test dataset:")
            logging.info(self.basic_inspection())

            # Step 2: Handle Duplicates and Missing Values
            
            self.train_set = self.handle_duplicate_values(self.train_set)
            self.test_set = self.handle_duplicate_values(self.test_set)
            logging.info("Handed duplicate values in train and test datasets.")
            logging.info(f"Duplicate Values in Trainset Now : {self.train_set.duplicated().sum()}")
            logging.info(f"Duplicate Values in Testset Now : {self.test_set.duplicated().sum()}")
            
            
            self.train_set = self.handle_missing_values(self.train_set)
            self.test_set = self.handle_missing_values(self.test_set)
            logging.info("Handed missing values in train and test datasets.")
            logging.info(f"Train dataset after missing value imputation:\n {self.train_set.head()}")
            logging.info(f"Test dataset after missing value imputation:\n {self.test_set.head()}")

            # Step 3: Handle Outliers
            self.train_set = self.handle_outliers(self.train_set)
            self.test_set = self.handle_outliers(self.test_set)
            logging.info("Handled outliers in train and test datasets.")
            logging.info(f"Train dataset after outlier handling:\n {self.train_set.head()}")
            logging.info(f"Test dataset after outlier handling:\n {self.test_set.head()}")

            # Step 4: Apply Feature Transformation Pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training data
            logging.info("Fitting and transforming training data using the preprocessor...")
            X_train_transformed_arr = preprocessor.fit_transform(self.train_set)
            logging.info(f"Transformed training data (first few rows):\n {pd.DataFrame(X_train_transformed_arr).head()}")

            # Transform the test data
            logging.info("Transforming test data using the preprocessor...")
            X_test_transformed_arr = preprocessor.transform(self.test_set)
            logging.info(f"Transformed test data (first few rows):\n {pd.DataFrame(X_test_transformed_arr).head()}")

            # Step 5: Saving the processed data as artifacts
            
            # Define the file paths for saving transformed data and preprocessor object
            transformed_train_file_path = os.path.join(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, "transformed_train_data.npz")
            transformed_test_file_path = os.path.join(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, "transformed_test_data.npz")
            transformed_object_file_path = os.path.join(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, "transformed_object.pkl")

            # Save the transformed training data
            self.data_transformation_config.UTILS.save_numpy_array_data(transformed_train_file_path, X_train_transformed_arr)
            logging.info(f"Saved transformed training data to {transformed_train_file_path}")

            # Save the transformed test data
            self.data_transformation_config.UTILS.save_numpy_array_data(transformed_test_file_path, X_test_transformed_arr)
            logging.info(f"Saved transformed test data to {transformed_test_file_path}")
        
            # Save the preprocessor (transformer object)
           
            joblib.dump(preprocessor, transformed_object_file_path)
            logging.info(f"Preprocessor object saved as a pickle file to {transformed_object_file_path}")

            # Step 6: Creating Data_Transformation_Artifacts object to return processed data paths
            data_transformation_artifacts = Data_Transformation_Artifacts(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                transformed_object_file_path=transformed_object_file_path
            )

            logging.info("Exited initiate_data_transformation method of Data_Transformation class")
            return data_transformation_artifacts

        except Exception as e:
            logging.error("Error in initiate_data_transformation", exc_info=True)
            raise CustomException(str(e), sys)
