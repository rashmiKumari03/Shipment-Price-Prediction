import os
import sys
import pickle
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from typing import Union, List
from io import StringIO
import boto3  # AWS SDK for Python to interact with AWS services like S3 for file storage.
from botocore.exceptions import ClientError  # To handle errors from AWS API calls made using boto3.
from mypy_boto3_s3.service_resource import Bucket  # Type hint for representing an S3 bucket resource.



class S3_Operation:
    
    
    """
    Class Name: S3_Operation
    Description: This class provides helper methods to interact with AWS S3 storage, 
                allowing operations like uploading files, checking for the existence of 
                a model in the bucket, and retrieving bucket objects.
    """

    def __init__(self):
        """
        Initializes the S3_Operation class.
        
        The constructor sets up a resource interface to interact with AWS S3 service.
        This is done using the boto3 library which allows Python applications to interact 
        with AWS services programmatically. The `resource` method creates a high-level 
        resource object for interacting with S3.
        """
        self.s3_resource = boto3.resource("s3")  # Establishes a resource object to interact with S3.

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Method Name: get_bucket
        Description: This method fetches the bucket object based on the provided bucket name.
                    The bucket object allows interactions like listing objects, checking object status, etc.
        Output: Returns the bucket object corresponding to the provided bucket name.
        """
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Successfully retrieved the bucket: %s", bucket_name)
            return bucket  # Return the S3 Bucket object
        except Exception as e:
            logging.error("Error while fetching bucket: %s", str(e))
            raise CustomException(str(e), sys)

    def is_model_present(self, bucket_name: str, s3_model_key: str) -> bool:
        """
        Method Name: is_model_present
        Description: This method checks if a model (or any file) is present in the specified S3 bucket 
                    under a specific key (like a file path). It validates the existence of the model.
        Output: Returns True if the model is found, False otherwise.
        
        Why it’s needed: 
        - In production systems, models and datasets are often stored on S3. 
        - Before attempting to download or update a model, it's crucial to verify its existence to avoid errors.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_model_key)]
            
            if len(file_objects) > 0:
                logging.info("Model is found in the S3 bucket: %s", bucket_name)
                return True
            else:
                logging.info("Model not found in the S3 bucket: %s", bucket_name)
                return False
        except Exception as e:
            logging.error("Error while checking model existence in S3: %s", str(e))
            raise CustomException(str(e), sys)

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Method Name: get_file_object
        Description: This method retrieves the file object for the given filename from the S3 bucket.
        Parameters:
        - filename: The name of the file to retrieve.
        - bucket_name: The S3 bucket name.
        Output: Returns the file object if found.
        """
        logging.info("Entered the get_file_object method of S3_Operation class")
        try:
            bucket = self.get_bucket(bucket_name)
            lst_objs = [object for object in bucket.objects.filter(Prefix=filename)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(lst_objs)
            logging.info("Exited the get_file_object method of S3_Operation class")
            return file_objs
        except Exception as e:
            logging.error("Error while retrieving file object: %s", str(e))
            raise CustomException(str(e), sys)

    def read_object(self, object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Method Name: read_object
        Description: This method reads an object from S3, optionally decoding it and making it readable.
        Parameters:
        - object_name: The object to read.
        - decode: Flag to determine if the object should be decoded (default is True).
        - make_readable: Flag to convert the object content into a StringIO object for easier handling.
        Output: Returns the object data either as a decoded string or as raw bytes, depending on the flags.
        """
        logging.info("Entered the read_object method of S3_Operation class")
        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )
            conv_func = lambda: StringIO(func()) if make_readable is True else func()
            logging.info("Exited the read_object method of S3_Operation class")
            return conv_func()
        except Exception as e:
            logging.error("Error while reading object from S3: %s", str(e))
            raise CustomException(str(e), sys)

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Method Name: load_model
        Description: This method loads a model file from S3 and deserializes it using pickle.
        Parameters:
        - model_name: The name of the model file to load.
        - bucket_name: The S3 bucket where the model is stored.
        - model_dir: (Optional) The directory where the model is stored within the bucket.
        Output: Returns the deserialized model object.
        """
        try:
            model_file = model_name if model_dir is None else model_dir + "/" + model_name
            f_obj = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(f_obj, decode=False)
            model = pickle.loads(model_obj)
            logging.info("Exited the load_model method of S3_Operation class")
            return model
        except Exception as e:
            logging.error("Error while loading model: %s", str(e))
            raise CustomException(str(e), sys)

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True) -> None:
        """
        Method Name: upload_file
        Description: This method uploads a file to an S3 bucket and optionally deletes it from the local system 
                    after the upload is successful.
        Parameters:
        - from_filename: Local file path that needs to be uploaded.
        - to_filename: The destination path in the S3 bucket where the file will be uploaded.
        - bucket_name: The S3 bucket where the file needs to be uploaded.
        - remove (optional): A boolean flag to remove the local file after successful upload. Default is True.
        """
        logging.info("Entered the upload_file method of S3_Operation class")
        try:
            logging.info(f"Uploading file '{from_filename}' to S3 bucket '{bucket_name}' with key '{to_filename}'")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            
            if remove:
                os.remove(from_filename)  # Deletes the local file after upload
                logging.info(f"File '{from_filename}' has been deleted from local storage after upload.")
            else:
                logging.info(f"File '{from_filename}' was not deleted from local storage.")
            
            logging.info("Successfully uploaded file to S3 and exited the upload_file method.")
        except Exception as e:
            logging.error("Error occurred while uploading file to S3: %s", str(e))
            raise CustomException(str(e), sys)
