import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Transformation_Artifacts , Model_Trainer_Artifacts , Model_Pusher_Artifacts)

from src.Shipment_Price_Prediction.entity.config_entity import Model_Pusher_Config

# Lets  Create first the Model_Pusher_Artifacts and Model_Pusher_Config


