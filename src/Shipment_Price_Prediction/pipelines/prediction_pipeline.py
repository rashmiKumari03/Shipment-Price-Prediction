import os
import sys
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np
from pandas import DataFrame

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation


class ShippingData:
    def __init__(
        self,
        line_item_quantity: int,
        pack_price: float,
        unit_price: float,
        weight: float,
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
        scheduled_delivery_date: Optional[datetime] = None,
        delivered_to_client_date: Optional[datetime] = None,
        delivery_recorded_date: Optional[datetime] = None,
        first_line_designation: Optional[str] = None,
    ):
        # Numerical
        self.line_item_quantity = line_item_quantity
        self.pack_price = pack_price
        self.unit_price = unit_price
        self.weight = weight
        self.unit_of_measure = unit_of_measure

        # Categorical
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

        # Dates
        self.scheduled_delivery_date = scheduled_delivery_date
        self.delivered_to_client_date = delivered_to_client_date
        self.delivery_recorded_date = delivery_recorded_date

        # Binary
        self.first_line_designation = first_line_designation

    def get_data(self) -> Dict:
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
                "Pack Price": [self.pack_price],
                "Unit Price": [self.unit_price],
                "Weight (Kilograms)": [self.weight],
                "Unit of Measure (Per Pack)": [self.unit_of_measure],
            }
            logging.info("Successfully created input data dictionary")
            return input_data
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def get_input_data_frame(self) -> DataFrame:
        try:
            input_dict = self.get_data()
            df = pd.DataFrame(input_dict)
            logging.info("Converted input data to DataFrame")
            return df
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)


class CostPredictor:
    def __init__(self):
        self.s3 = S3_Operation()
        self.bucket_name = BUCKET_NAME
        self.log_file = os.path.join("Artifacts", "Log_Prediction", "prediction_logs.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_prediction(self, X: DataFrame, prediction: float):
        try:
            log_data = X.copy()
            log_data["Predicted Cost"] = prediction
            log_data["Timestamp"] = datetime.now()

            if not os.path.exists(self.log_file):
                log_data.to_csv(self.log_file, index=False)
            else:
                log_data.to_csv(self.log_file, mode="a", index=False, header=False)

            logging.info("Prediction logged successfully")
        except Exception as e:
            logging.error(f"Logging failed: {e}")

    def predict(self, X: DataFrame) -> float:
        try:
            best_model = self.s3.load_model(MODEL_FILE_NAME, self.bucket_name)
            logging.info("Loaded the best model from S3 bucket")

            logging.info(f"Columns in input X: {X.columns.tolist()}")
            result = best_model.prediction(X)

            # Convert log1p predictions back to original scale
            final_cost_result = np.expm1(result)

            logging.info(f"Prediction result (original scale): {final_cost_result}")

            # Log prediction
            self.log_prediction(X, float(final_cost_result[0]))

            # Return safely
            if isinstance(final_cost_result, (list, np.ndarray)) and len(final_cost_result) > 0:
                return float(final_cost_result[0])
            elif isinstance(final_cost_result, float):
                return final_cost_result
            else:
                logging.warning("Prediction result is empty or not in expected format")
                return None

        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
