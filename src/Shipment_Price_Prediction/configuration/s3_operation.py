import os
import sys
import pickle
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

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
            # Accessing the bucket using S3 resource API
            bucket = self.s3_resource.Bucket(bucket_name)  
            logging.info("Successfully retrieved the bucket: %s", bucket_name)
            return bucket  # Return the S3 Bucket object
        except Exception as e:
            # Logs any exceptions raised during the operation and raises a custom exception.
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
            # Retrieves the bucket object
            bucket = self.get_bucket(bucket_name)
            # Filters the objects in the bucket based on the provided prefix (s3_model_key)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_model_key)]
            
            # If there are any objects found with the provided key, return True, otherwise False.
            if len(file_objects) > 0:
                logging.info("Model is found in the S3 bucket: %s", bucket_name)
                return True
            else:
                logging.info("Model not found in the S3 bucket: %s", bucket_name)
                return False
        except Exception as e:
            logging.error("Error while checking model existence in S3: %s", str(e))
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
        
        Why it's needed:
        - In production scenarios, large files, models, and datasets are frequently uploaded to cloud storage like S3 for accessibility, backup, and scalability.
        - The remove flag helps in cleaning up local storage after uploading files to avoid unnecessary clutter.
        """
        logging.info("Entered the upload_file method of S3_Operation class")
        try:
            logging.info(f"Uploading file '{from_filename}' to S3 bucket '{bucket_name}' with key '{to_filename}'")
            
            # Uploading the file using the boto3 client's upload_file method
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            
            # Check if the file should be removed from local storage after upload
            if remove:
                os.remove(from_filename)  # Deletes the local file
                logging.info(f"File '{from_filename}' has been deleted from local storage after upload.")
            else:
                logging.info(f"File '{from_filename}' was not deleted from local storage.")
            
            logging.info("Successfully uploaded file to S3 and exited the upload_file method.")
        except Exception as e:
            # Logs the error and raises a custom exception if something goes wrong
            logging.error("Error occurred while uploading file to S3: %s", str(e))
            raise CustomException(str(e), sys)
