from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import sys
from typing import Dict, Optional
import pandas as pd
from pandas import DataFrame

from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation

class ShippingData:
    def __init__(self,
                 line_item_quantity,
                 line_item_value,
                 pack_price,
                 unit_price,
                 weight,
                 country,
                 shipment_mode,
                 scheduled_delivery_date,
                 delivered_to_client_date,
                 first_line_designation):
        
        self.line_item_quantity = line_item_quantity
        self.line_item_value = line_item_value
        self.pack_price = pack_price
        self.unit_price = unit_price
        self.weight = weight
        self.country = country
        self.shipment_mode = shipment_mode
        self.scheduled_delivery_date = scheduled_delivery_date
        self.delivered_to_client_date = delivered_to_client_date
        self.first_line_designation = first_line_designation
        
    def get_data(self) -> Dict:
        """
        Method Name : get_data
        Description : This method gathers input data into a dictionary
        Output : Input data in dictionary format
        """
        logging.info("Entered the get_data method of ShippingData class")
        try:
            input_data = {
                "Line_Item_Quantity": [self.line_item_quantity],
                "Line_Item_Value": [self.line_item_value],
                "Pack_Price": [self.pack_price],
                "Unit_Price": [self.unit_price],
                "Weight": [self.weight],
                "Country": [self.country],
                "Shipment_Mode": [self.shipment_mode],
                "Scheduled_Delivery_Date": [self.scheduled_delivery_date],
                "Delivered_To_Client_Date": [self.delivered_to_client_date],
                "First_Line_Designation": [self.first_line_designation]
            }
            logging.info("Successfully created input data dictionary")
            return input_data
        
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        
    def get_input_data_frame(self) -> DataFrame:
        """
        Method Name : get_input_data_frame
        Description : This method converts the input data dictionary into a DataFrame
        Output : DataFrame
        """
        logging.info("Entered the get_input_data_frame method of ShippingData class")
        try:
            input_dict = self.get_data()
            logging.info("Converted input data to dictionary")
            return pd.DataFrame(input_dict)
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        
class CostPredictor:
    def __init__(self):
        self.s3 = S3_Operation()
        self.bucket_name = BUCKET_NAME
        
    def predict(self, X: DataFrame) -> float:
        """
        Method Name : predict
        Description : This method predicts shipment cost using the best model
        Output : Prediction
        """
        logging.info("Entered the predict method of CostPredictor class")
        try:
            best_model = self.s3.load_model(MODEL_FILE_NAME, self.bucket_name)
            logging.info("Loaded the best model from S3 bucket")
            result = best_model.predict(X)
            logging.info("Successfully made predictions")
            return result[0] if result else None
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
