import sys
import pandas as pd
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from json import loads
from pymongo import MongoClient
from pymongo.database import Database
from pandas import DataFrame
from typing import Collection

from src.Shipment_Price_Prediction.constant import DB_URL



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
        logging.info("Entered get_database method of MongoDB_Operation class ")
        
        try:
            # Getting the DB
            db = self.client[db_name]
            
            logging.info(f"Created {db_name} database in MongoDB")
            logging.info("Exited get_database method of MongoDB_Operation class")
            
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
        logging.info("Entered get_collection method of MongoDB_Operation class")
        
        try:
            # Getting  the collection name
            collection = database[collection_name]
            
            logging.info(f"Created {collection_name} collection in MongoDB")
            logging.info("Exited get_collection method of MongoDB_Operation class")
            
            return collection
        
        except Exception as e :
            raise CustomException(e,sys)
        
        
    def get_collection_as_dataframe(self,db_name,collection_name) -> DataFrame:
        """
        Method Name: get_collection_as_dataframe
        
        Description : This method is used for converting the selected collection to dataframe
        
        Output : A collectionis retured from the selected db_name and collection_name
        
        """
        logging.info("Entered get_collection_as_dataframe method of MongoDB_Operation class")
            
        try:
            # Getting the database
            database = self.get_database(db_name)
            
            # Getting the collection name
            collection = database.get_collection(name=collection_name) 
            
            # Reading the dataframe and dropping the _id column , Because _id is the id in mongodb for each record
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
                
            logging.info("Converted collection to Dataframe")
            logging.info("Exited get_collection_as_dataframe method of MongoDB_Operation class")
            
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
   
   def insert_dataframe_as_record(self,dataframe,db_name,collection_name) -> None:
       """
       Method Name : insert_dataframe_as_record 
       
       Description : This method inserts the dataframe as record in database collection
       
       Output : The dataframe is inserted in database collection
       """
       
       logging.info("Entered insert_dataframe_as_record method of MongoDB_Operation class")
       
       try:
           # Converting dataframe into json
           records = loads(dataframe.T.to_json()).values()
           logging.info(f"Converted dataframe to json records")
           
           # Getting the database and collection
           database = self.get_database(db_name)
           collection = database.get_collection(collection_name)
           logging.info("Inserting records to MongoDB")
           
           # Inserting data to MongoDB database
           collection.insert_many(records)
           logging.info("Inserted records to MongoDB")
           logging.info("Exited the insert_dataframe_as_record method of MongoDB_Operation class")
           
       except Exception as e:
           raise CustomException(e,sys)
       
       
           
           
        
            
        
        
        
        
        