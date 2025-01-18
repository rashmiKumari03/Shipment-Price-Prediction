
import os
import sys
import pickle
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from pandas import DataFrame,read_csv
from typing import List , Union
from src.Shipment_Price_Prediction.constant import *

from io import StringIO   # For handling string data as file-like objects
import boto3              # AWS SDK (Software Development Kit) for Python to interact with AWS services like S3 for file storage.
from botocore.exceptions import ClientError   # To handle errors from AWS API calls made using boto3.
from mypy_boto3_s3.service_resource import Bucket  # Type hint for representing an S3 bucket resource, improving code clarity and IDE support.


class S3_Operation:
    
    def __init__(self):
     
        self.s3_resource = boto3.resource("s3")
        
        
    def upload_file(self,
                    from_filename: str,
                    to_filename: str,
                    bucket_name: str,
                    remove: bool = True) -> None:
        logging.info("Entered the upload_file method of S3_Operation class")
        try:
            logging.info(f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket")
            self.s3_resource.meta.client.upload_file(from_filename,bucket_name,to_filename)  
            
            if remove is True:
                os.remove(from_filename)
                logging.info(f"Remove is set to {remove} , deleted the file")
            else:
                logging.info(f"Remove is set to {remove}, not deleted the file")
                
            logging.info("Exited the upload_file method of S3_Operation class")
        
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    
        