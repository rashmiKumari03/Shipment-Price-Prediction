import sys
from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.entity.config_entity import S3_to_Local_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import S3_to_Local_Artifacts
from src.Shipment_Price_Prediction.configuration.s3_operation import S3_Operation

import warnings
warnings.filterwarnings('ignore')

class S3_to_Local_Model:
    """
    The S3_to_Local_Model class is responsible for pulling the
    best model from the S3 bucket and saving it locally for
    DVC tracking or re-deployment.

    Attributes:
    - s3_to_local_config: Config class with S3 details and local directory
    - s3: S3_Operation object to handle S3 operations
    """

    def __init__(
        self,
        s3_to_local_config: S3_to_Local_Config,
        s3: S3_Operation
    ):
        self.s3_to_local_config = s3_to_local_config
        self.s3 = s3

    def initiate_s3_to_local(self) -> S3_to_Local_Artifacts:
        """
        Downloads the best model from S3 to a local path for DVC tracking.
        """
        logging.info("Entered the initiate_s3_to_local method of S3_to_Local_Model class")
        try:
            local_download_path = self.s3.download_model_from_s3(
                bucket_name=self.s3_to_local_config.BUCKET_NAME,
                s3_model_key=self.s3_to_local_config.S3_MODEL_KEY_PATH,
                local_dir=self.s3_to_local_config.LOCAL_MODEL_SAVE_DIR
            )
            logging.info(f"Best model successfully downloaded from S3 to local path: {local_download_path}")

            artifact = S3_to_Local_Artifacts(
                bucket_name=self.s3_to_local_config.BUCKET_NAME,
                s3_model_path=self.s3_to_local_config.S3_MODEL_KEY_PATH,
                local_model_path=local_download_path
            )

            logging.info("Exited the initiate_s3_to_local method of S3_to_Local_Model class")
            return artifact

        except Exception as e:
            logging.error(f"Error in initiate_s3_to_local: {e}")
            raise CustomException(str(e), sys)



if __name__ == "__main__":
    try:
        logging.info("*******************")
        logging.info(">>>>>> S3 to Local stage started <<<<<<")

        # ------------------------------------------------------
        # Creating the S3 to Local Config
        # ------------------------------------------------------
        s3_to_local_config = S3_to_Local_Config()

        # ------------------------------------------------------
        # Instantiating S3_Operation
        # ------------------------------------------------------
        s3 = S3_Operation()

        # ------------------------------------------------------
        # Instantiating the S3_to_Local_Model class
        # ------------------------------------------------------
        s3_to_local_model = S3_to_Local_Model(
            s3_to_local_config=s3_to_local_config,
            s3=s3
        )

        # ------------------------------------------------------
        # Calling the S3 to Local pipeline
        # ------------------------------------------------------
        s3_to_local_artifact = s3_to_local_model.initiate_s3_to_local()

        logging.info(">>>>>> S3 to Local stage completed <<<<<<\n")
        logging.info(f"S3 to Local Artifacts: {s3_to_local_artifact}")

    except Exception as e:
        logging.exception(e)
        raise e
