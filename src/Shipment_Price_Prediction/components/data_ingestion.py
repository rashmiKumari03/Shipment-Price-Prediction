
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




