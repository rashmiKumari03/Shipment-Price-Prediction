import shutil
import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

import xgboost
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators
from yaml import safe_dump

from typing import Dict,Tuple,List
import dill
import yaml

from src.Shipment_Price_Prediction.constant import *


class MainUtils:
    def read_yaml_file(self,filename:str) -> dict:
        logging.info("Entered the read_yaml_file method of MainUtils class")
        
        try:
            with open(filename,"r") as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys) 

    def write_json_to_yaml_file(self,json_file:dict , yaml_file_path: str) -> yaml :
        logging.info("Entered the write_json_to_yaml_file method of MainUtils class")
        try:
            data = json_file
            stream = open(yaml_file_path,"w")
            yaml.dump(data,stream)
        
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        
        
    def save_numpy_array_data(self , file_path:str , array:np.array):
        logging.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            with open(file_path , "wb") as file_obj:
                np.save(file_obj,array)
            logging.info("Exited the save_numpy_array_data method of MainUtils class")
            
            return file_path
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        
        
    def load_numpy_array_data(self,file_path:str) -> np.array:
        logging.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path,"rb") as file_obj:
                
                return np.load(file_obj)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        
    def get_tunned_model(self,
                         model_name:str,
                         train_x:DataFrame,
                         train_y : DataFrame,
                         test_x : DataFrame,
                         test_y : DataFrame) -> Tuple[float,object,str]:
        logging.info("Entered the get_tuned_model method of MainUtils class")
        try:
            model = self.get_base_model(model_name)
            model_best_params = self.get_model_params(model,train_x,train_y)
            model.set_params(**model_best_params)
            model.fit(train_x,train_y)
            preds = model.predict(test_x)
            model_score = self.get_model_score(test_y,preds)
            logging.info("Exited the get_tunned_model method of MainUtils class")
            
            return model_score , model , model.__class__.__name__
        
        
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        
              


