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


    def basic_inspection(self, data:DataFrame):
        
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
        logging.info("Entered segregate_datatype method of Data_Transformation class")
        try:
            # Segregate columns by type using SCHEMA_CONFIG
            numerical_cols = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            categorical_cols = self.data_transformation_config.SCHEMA_CONFIG["categorical_columns"]
            datetime_cols = self.data_transformation_config.SCHEMA_CONFIG["datetime_columns"]

            # Convert numerical columns
            if numerical_cols:
                data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')
                logging.info(f"Converted {len(numerical_cols)} numerical columns to numeric type.")

            # Convert categorical columns
            if categorical_cols:
                data[categorical_cols] = data[categorical_cols].astype("string")
                logging.info(f"Converted {len(categorical_cols)} categorical columns to string type.")

            # Convert datetime columns
            if datetime_cols:
                data[datetime_cols] = data[datetime_cols].apply(pd.to_datetime, errors='coerce')
                logging.info(f"Converted {len(datetime_cols)} datetime columns to datetime type.")

            logging.info("Data types have been successfully converted.")
            logging.info("Exited segregate_datatype method of Data_Transformation class")
            return data

        except Exception as e:
            logging.exception(f"Error in segregate_datatype method:{CustomException(str(e),sys)}")
            raise CustomException(str(e),sys)

        

    def segregate_datatype(self, data: DataFrame) -> DataFrame:
        """
        Method Name : segregate_datatype

        Description : This method segregates and converts columns in the DataFrame to their appropriate data types.
                    It ensures that numerical columns are of numeric type, categorical columns are converted to
                    categorical type, and datetime columns are appropriately parsed.

        Output      : DataFrame with columns converted to the appropriate data types.
        """
        logging.info("Entered segregate_datatype method of Data_Transformation class")
        try:
            # Segregate columns by type
            numerical_cols = data.select_dtypes(include=['number']).columns
            categorical_cols = data.select_dtypes(include=['object','string']).columns
            datetime_cols = data.select_dtypes(include=['datetime']).columns

            # Convert numerical columns
            if not numerical_cols.empty:
                data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')
                logging.info(f"Converted {len(numerical_cols)} numerical columns to numeric type.")

            # Convert categorical columns
            if not categorical_cols.empty:
                data[categorical_cols] = data[categorical_cols].astype("string")
                logging.info(f"Converted {len(categorical_cols)} categorical columns to string type.")

            # Convert datetime columns
            if not datetime_cols.empty:
                data[datetime_cols] = data[datetime_cols].apply(pd.to_datetime, errors='coerce')
                logging.info(f"Converted {len(datetime_cols)} datetime columns to datetime type.")

            logging.info("Data types have been successfully converted.")
            logging.info("Exited segregate_datatype method of Data_Transformation class")
            return data

        except Exception as e:
            error_message = f"Error in segregate_datatype method: {str(e)}"
            logging.error(error_message)
            raise CustomException(error_message, sys)
        


        
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
            # Get columns from SCHEMA_CONFIG
            numerical_cols = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            categorical_cols = self.data_transformation_config.SCHEMA_CONFIG["categorical_columns"]
            datetime_cols = self.data_transformation_config.SCHEMA_CONFIG["datetime_columns"]

            # Impute missing values in numerical columns
            numerical_imputer = SimpleImputer(strategy='median')
            data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
            logging.info(f"Imputed missing values in numerical columns: {numerical_cols}")

            # Impute missing values in categorical columns
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
            logging.info(f"Imputed missing values in categorical columns: {categorical_cols}")

            # Impute missing values in datetime columns
            if datetime_cols:
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
        
        Description : 
                    This method returns a preprocessor object that applies various data transformations such as 
                    imputation, encoding, and scaling to the dataset. The method builds a transformation pipeline 
                    using SimpleImputer, OneHotEncoder, and BinaryEncoder to process different feature types.
                    
                    This method creates a preprocessor (transformer) that:
                        - Drops irrelevant columns.
                        - Fills missing values.
                        - Scales numerical data.
                        - Encodes categorical data.
                        - Transforms datetime columns into useful features.

        Output      : Preprocessor object (ColumnTransformer) for transforming the dataset.
        """
        
        logging.info("Entered get_data_transformer_object method of Data_Transformation class")

        try:
            # Get the column names from the configuration
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG["numerical_columns"]
            categorical_columns = self.data_transformation_config.SCHEMA_CONFIG["categorical_columns"]
            binary_columns = self.data_transformation_config.SCHEMA_CONFIG["binary_columns"]
            datetime_columns = self.data_transformation_config.SCHEMA_CONFIG["datetime_columns"]
            irrelevant_columns_to_drop = self.data_transformation_config.SCHEMA_CONFIG["drop_columns"]

            # Define how to process numerical columns: impute missing values and scale them
            numeric_imputer = SimpleImputer(strategy="median")  # Fill missing values with the median value
            numeric_transformer = Pipeline(steps=[
                ('imputer', numeric_imputer),
                ('scaler', StandardScaler())  # Scale the data to have mean=0 and variance=1
            ])

            # Define how to process categorical columns: impute missing values and apply one-hot encoding
            categorical_imputer = SimpleImputer(strategy="most_frequent")  # Fill missing values with the most frequent value
            categorical_transformer = Pipeline(steps=[
                ('imputer', categorical_imputer),
                ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore"))  # One-hot encode the categorical data
            ])

            # Define how to process binary columns: apply binary encoding
            binary_transformer = Pipeline(steps=[
                ('binary_encoder', BinaryEncoder())
            ])

            # Define how to process datetime columns: extract useful datetime features
            def extract_datetime_features(df):
                for col in datetime_columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_weekday'] = df[col].dt.weekday
                return df.drop(columns=datetime_columns)  # Drop original datetime columns

            datetime_transformer = FunctionTransformer(
                lambda df: extract_datetime_features(df),
                validate=False
            )

            # Drop irrelevant columns
            drop_transformer = FunctionTransformer(
                lambda x: x.drop(columns=irrelevant_columns_to_drop, errors='ignore'),
                validate=False
            )

            # Create a ColumnTransformer to apply different transformations to different columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('drop_irrelevant_columns', drop_transformer, irrelevant_columns_to_drop),  # Drop irrelevant columns first
                    ('numerical', numeric_transformer, numerical_columns),  # Process numerical columns
                    ('categorical', categorical_transformer, categorical_columns),  # Process categorical columns
                    ('binary', binary_transformer, binary_columns),  # Process binary columns
                    ('datetime', datetime_transformer, datetime_columns)  # Transform datetime columns
                ]
            )

            logging.info("Data transformer object created successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in creating data transformer object: {str(e)}")
            raise CustomException(f"Error in creating data transformer object: {str(e)}", sys)


    def initiate_data_transformation(self) -> Data_Transformation_Artifacts:
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

            # Step 0: Basic Inspection of the Train and Test sets
            logging.info("Basic Inspection of Train dataset:")
            logging.info(self.basic_inspection(self.train_set))
            
            logging.info("Basic Inspection of Test dataset:")
            logging.info(self.basic_inspection(self.test_set))
            
            # Step 1: Segregate the Data based on datatype (This will be schema-aware if needed)
            self.train_set = self.segregate_datatype(self.train_set)
            self.test_set = self.segregate_datatype(self.test_set)
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
            logging.error(f"Error in apply_transformations method: {str(e)}")
            raise CustomException(str(e), sys)
