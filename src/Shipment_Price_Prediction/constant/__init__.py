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

TARGET_COLUMN = "Shipment_Cost"

# Split ratio for dividing the dataset into training and testing sets
TEST_SIZE = 0.2  # Proportion of data to be used for testing during train-test split

ARTIFACTS_DIR = os.path.join(from_root(),"artifacts",TIMESTAMP)


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
TRANSFORMED_TRAIN_DATA_DIR = "Transformed_Train"
TRANSFORMED_TEST_DATA_DIR = "Transformed_Test"
TRANSFORMED_TRAIN_DATA_FILE_NAME = "Transformed_train_data.npz"
TRANSFORMED_TEST_DATA_FILE_NAME = "Transformed_test_data.npz"
PREPROCESSOR_OBJECT_FILE_NAME = "shipping_preprocessor.pkl"
