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
# Then we will create the config_entity... in 


