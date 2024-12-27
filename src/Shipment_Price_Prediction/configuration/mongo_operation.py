import sys
from json import loads
from typing import Collection
from pandas import DataFrame
from pymongo.database import Database
from pymongo import MongoClient
import pandas as pd


from src.Shipment_Price_Prediction.constant import DB_URL
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException


# Initializing logger

logger = logging.getLogger(__name__)


class MongoDB_Operation:
    def __init__(self):
        self.DB_URL = DB_URL
        self.client = MongoClient(self.DB_URL)
        
        
    def get_database(self,db_name) -> Database:
        """
        Method Name : get_database
        
        Description : This method gets database from MongoDB from the db_name
        
        Output : A database is created in MongoDB with name as db_name
        """
        logger.info("Entered get_database method of MongoDB_Operation class ")
        
        try:
            # Getting the DB
            db = self.client[db_name]
            
            logger.info(f"Created {db_name} database in MongoDB")
            logger.info("Exited get_database method of MongoDB_Operation class")
            
            return db
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    """
      get_database: Not a @staticmethod because it uses self.client, an instance attribute initialized in __init__.
      get_collection: Used @staticmethod because it doesn't rely on instance (self) or class (cls) attributes.
    """
    
    @staticmethod  # By making get_collection a @staticmethod, we also allow it to be called directly on the class without requiring an instance, as it doesn’t rely on instance data.  
    def get_collection(database,collection_name) -> Collection:
        """
        Method Name : get_collection
        
        Description : This method gets collection from the particular database and collection name.
        
        Output : A collection is returned from database with name as collection name
        """
        logger.info("Entered get_collection method of MongoDB_Operation class")
        
        try:
            # Getting  the collection name
            collection = database[collection_name]
            
            logger.info(f"Created {collection_name} collection in MongoDB")
            logger.info("Exited get_collection method of MongoDB_Operation class")
            
            return collection
        except Exception as e :
            raise CustomException(e,sys)
            
        
        
        
        
        
        