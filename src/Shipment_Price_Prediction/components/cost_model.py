import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

import warnings
warnings.filterwarnings("ignore")


class Cost_Model:

    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.model = trained_model_object
        

    def prediction(self, X:DataFrame):
        """
        Predict shipment costs based on input features.
        """
        logging.info("Entered the 'prediction' method of the Cost_Model class.")
        try:
    
            print(X.head())
            logging.info(f"X looks like : \n {X.head()}")
            # Preprocess the input data
            
            # Loading the preprocessor.pkl
            logging.info(f"Preprocessor and Model object provided in the predict..")
            
            logging.info(f"preprocessor_obj is :{self.preprocessing_object}")
            X_transformed = X.drop_duplicates()
            X_transformed = self.preprocessing_object.transform(X)
            # Make predictions using the trained model
            logging.info("Transformation on new/test data has Completed!")
            return self.model.predict(X_transformed)
            

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(str(e), sys)

    def __repr__(self):
        return f"{type(self.model).__name__}()"

    def __str__(self):
        return f"{type(self.model).__name__}()"
