from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import sys
from typing import Dict, Optional
import pandas as pd
import numpy as np
from pandas import DataFrame

from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation


class ShippingData:
    
    def __init__(self,
                 line_item_quantity: int,
                 line_item_value: float,
                 pack_price: float,
                 unit_price: float,
                 weight: float,
                 freight_cost: Optional[float] = None,
                 line_item_insurance: Optional[float] = None,
                 unit_of_measure: Optional[str] = None,
                 country: str = None,
                 managed_by: Optional[str] = None,
                 fulfill_via: Optional[str] = None,
                 vendor_inco_term: Optional[str] = None,
                 shipment_mode: Optional[str] = None,
                 product_group: Optional[str] = None,
                 sub_classification: Optional[str] = None,
                 vendor: Optional[str] = None,
                 molecule_test_type: Optional[str] = None,
                 brand: Optional[str] = None,
                 dosage: Optional[str] = None,
                 dosage_form: Optional[str] = None,
                 manufacturing_site: Optional[str] = None,
                 scheduled_delivery_date: datetime = None,
                 delivered_to_client_date: Optional[datetime] = None,
                 delivery_recorded_date: Optional[datetime] = None,
                 first_line_designation: Optional[str] = None):

                # Numerical Inputs (8)
                self.line_item_quantity = line_item_quantity
                self.line_item_value = line_item_value
                self.pack_price = pack_price
                self.unit_price = unit_price
                self.weight = weight
                self.freight_cost = freight_cost
                self.line_item_insurance = line_item_insurance
                self.unit_of_measure = unit_of_measure

                # Categorical Inputs (14)
                self.country = country
                self.managed_by = managed_by
                self.fulfill_via = fulfill_via
                self.vendor_inco_term = vendor_inco_term
                self.shipment_mode = shipment_mode
                self.product_group = product_group
                self.sub_classification = sub_classification
                self.vendor = vendor
                self.molecule_test_type = molecule_test_type
                self.brand = brand
                self.dosage = dosage
                self.dosage_form = dosage_form
                self.manufacturing_site = manufacturing_site

                # Date Inputs (3)
                self.scheduled_delivery_date = scheduled_delivery_date
                self.delivered_to_client_date = delivered_to_client_date
                self.delivery_recorded_date = delivery_recorded_date

                # Binary Input (1)
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
                "Country": [self.country],
                "Managed By": [self.managed_by],
                "Fulfill Via": [self.fulfill_via],
                "Vendor INCO Term": [self.vendor_inco_term],
                "Shipment Mode": [self.shipment_mode],
                "Product Group": [self.product_group],
                "Sub Classification": [self.sub_classification],
                "Vendor": [self.vendor],
                "Molecule/Test Type": [self.molecule_test_type],
                "Brand": [self.brand],
                "Dosage": [self.dosage],
                "Dosage Form": [self.dosage_form],
                "Manufacturing Site": [self.manufacturing_site],
                "First Line Designation": [self.first_line_designation],

                "Scheduled Delivery Date": [self.scheduled_delivery_date],
                "Delivered to Client Date": [self.delivered_to_client_date],
                "Delivery Recorded Date": [self.delivery_recorded_date],

                "Line Item Quantity": [self.line_item_quantity],
                "Line Item Value": [self.line_item_value],
                "Pack Price": [self.pack_price],
                "Unit Price": [self.unit_price],
                "Weight (Kilograms)": [self.weight],
                "Freight Cost (USD)": [self.freight_cost],
                "Line Item Insurance (USD)": [self.line_item_insurance],
                "Unit of Measure (Per Pack)": [self.unit_of_measure]
            }
            
            logging.info("Successfully created input data dictionary")
            return input_data

        except Exception as e:  # Correct indentation
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
            
            # Check the columns of X
            logging.info(f"Columns in X: {X.columns.tolist()}")
            result = best_model.prediction(X)
            final_cost_result = np.expm1(result)
            logging.info("Successfully made predictions")
            logging.info(f"Result is based on prediction in model_predictor: {final_cost_result}")
            
            # Check if result is a list or array-like
            if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                return result[0]
            elif isinstance(result, float):  # If result is a single float value
                return result
            else:
                logging.warning("Prediction result is empty or not in expected format.")
                return None
        
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
