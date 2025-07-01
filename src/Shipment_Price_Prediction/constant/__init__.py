# Libraries required for the data ingestion process
# The `constant/__init__.py` file is primarily used to define static configurations and constants, 
# such as file paths, database URLs, and reusable parameters.

import os
from datetime import datetime
from os import environ
from from_root.root import from_root

# Timestamp: This will be used to track when artifacts are created in various pipeline steps.
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Configuration file paths
MODEL_CONFIG_FILE = "config/model.yaml"  # Path to the MongoDB configuration file
SCHEMA_FILE_PATH = "config/schema.yaml"  # Path to the schema validation file

# Database configuration
DB_URL = environ.get("MONGO_DB_URL")  # MongoDB connection URL is fetched from the environment variable.
                                      # We will make sure to set this in our environment variables.

DB_NAME = "Shipment"                  # Name of the database
COLLECTION_NAME = "Shipping data"     # Name of the collection within the database

"""
Target column for the prediction task
Note: The 'Shipment_Cost' column is derived during feature engineering using the following columns:
        - 'Freight Cost (USD)'
        - 'Line Item Value'
        - 'Line Item Insurance (USD)'

"""

TARGET_COLUMN = "Shipment Price"

# Split ratio for dividing the dataset into training and testing sets
TEST_SIZE = 0.2  # Proportion of data to be used for testing during train-test split


ARTIFACTS_DIR = os.path.join(from_root(),"Artifacts")


# Data Ingestion related COnstant.
DATA_INGESTION_ARTIFACTS_DIR = "Data_Ingestion_Artifacts"
DATA_INGESTION_TRAIN_DIR = "Train"
DATA_INGESTION_TEST_DIR = "Test"
DATA_INGESTION_TRAIN_FILE_NAME = "train.csv"
DATA_INGESTION_TEST_FILE_NAME =  "test.csv"

# Now creating the configuration ---> Go to entity folder and there --> config_entity.py


# Data Validation related Constants:

DATA_VALIDATION_ARTIFACT_DIR = "Data_Validation_Artifacts"
DATA_DRIFT_FILE_NAME = "Data_Drift_Report.yaml"


# Data Transformation related Constants:
DATA_TRANSFORMATION_ARTIFACTS_DIR = "Data_Transformation_Artifacts"
TRAIN_DATA_FILE_NAME = "train_data.npz"
TEST_DATA_FILE_NAME = "test_data.npz"


PREPROCESSOR_OBJECT_FILE_NAME = "shipping_preprocessor.pkl"   # used for predicting new data


# Model Trainer related Constants:
MODEL_TRAINER_ARTIFACTS_DIR = "Model_Trainer_Artifacts"
MODEL_FILE_NAME = "Shipping_Price_Prediction_Model.pkl"
MODEL_SAVE_FORMAT = ".pkl"


# Constants related to S3 Bucket 
BUCKET_NAME = "shipment-price-predictor-ml"   # All must be small always otherwise s3 won't accept it
S3_MODEL_NAME = "Shipping_Price_Prediction_Model.pkl"  # Both Names Model_filename and S3_Modelname must be same to fetch it from s3 later.

# S3 to Local
S3_TO_LOCAL = "S3_Best_Model_Artifacts"

# Constants for Webpage
APP_HOST = "127.0.0.1"
APP_PORT = 8080

