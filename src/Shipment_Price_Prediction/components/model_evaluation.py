import os 
import sys
import pandas as pd
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts,Model_Trainer_Artifacts,Model_Evaluation_Artifacts)
from src.Shipment_Price_Prediction.entity.config_entity import Model_Evaluation_Config

# Define constants for model evaluation in the constants folder.
# Implement the S3 bucket configuration in the configuration folder for better organization.
