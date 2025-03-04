import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException

from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts, Model_Trainer_Artifacts, Model_Evaluation_Artifacts)
from src.Shipment_Price_Prediction.entity.config_entity import Model_Evaluation_Config

# Define constants for model evaluation in the constants folder.
# Implement the S3 bucket configuration in the configuration folder for better organization.

"""
The presence of two models (S3 model and local/trained model) in the code reflects a comparison mechanism used to determine whether a newly trained model
(local model) is better than the currently deployed model (S3 model).

Here's the purpose of having these two models:

Why Two Models?
1. S3 Model (Deployed Model):
   - This represents the model currently deployed in production.
   - It is fetched from an S3 bucket to evaluate its performance on the latest data.

2. Local/Trained Model:
   - This is the newly trained model, which is evaluated against the same dataset to determine if it performs better than the current production model.

The Purpose of Comparison:
The goal is to:
- Compare the performance (using an evaluation metric like R² score) of the newly trained model against the S3 model.
- Decide whether to replace the S3 model with the newly trained one, depending on which performs better on the test dataset.

How It Works in Your Code:
1. Trained Model:
   - You load the newly trained model from a file path and use it to make predictions on the test dataset (`y_hat_trained_model`).
   - The performance is measured using the R² score (`trained_model_r2_score`).

2. S3 Model:
   - The `get_s3_model()` method fetches the model from the S3 bucket, if it exists.
   - Predictions are made using this model (`y_hat_s3_model`), and its performance is also measured using the R² score (`s3_model_r2_score`).

3. Comparison:
   - If the newly trained model's R² score is better than the S3 model's R² score, the new model is considered better and may replace the S3 model.
   - The difference in scores is calculated and logged (`difference`).

What We Need to Decide:
1. Is Comparing the Models Necessary?
   - Yes, if we want to ensure only the best-performing model is used in production.
   - No, if we always want to deploy the newly trained model without comparison.

2. Should the S3 Model Always Exist?
   - If the S3 model doesn't exist, the code currently defaults to using the newly trained model without comparison.
"""

# This class elements were logged.
@dataclass
class Evaluate_Model_Response:
    trained_model_r2_score: float
    s3_model_r2_score: float
    is_model_accepted: bool
    difference: float


class Model_Evaluation:
    
    def __init__(self, model_trainer_artifact: Model_Trainer_Artifacts, model_evaluation_config: Model_Evaluation_Config, data_ingestion_artifact: Data_Ingestion_Artifacts):
        
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        
        if model_trainer_artifact is None or model_trainer_artifact.trained_model_file_path is None:
             raise CustomException("Model_Trainer_Artifacts is not properly initialized. Ensure trained_model_file_path is provided.", sys)

    def get_s3_model(self) -> object:
        """ 
        Method Name : get_s3_model
        Description : This method gets model from s3 bucket.
        Output      : Model
        """
        logging.info("Entered the get_s3_model method of the Model_Evaluation class")
        try:
            # Checking whether model is present in the S3 bucket or not?
            status = self.model_evaluation_config.S3_Operation.is_model_present(BUCKET_NAME, S3_MODEL_NAME)
            logging.info(f"Got the status: is model present? => {status}")

            # If model is present, load the model
            if status == True:
                model = self.model_evaluation_config.S3_Operation.load_model(MODEL_FILE_NAME, BUCKET_NAME)
                logging.info("Exited the get_s3_model method of Model_Evaluation class")
                
                return model
            else:
                logging.info("Model Not Found in S3!")
                
                return None
            
        except Exception as e:
            logging.info(f"Error occurred: {str(e)}")
            raise CustomException(str(e), sys)

    def evaluate_model(self) -> Evaluate_Model_Response:
        """ 
        Method Name : evaluate_model
        Description : This method evaluates the S3 bucket model and the newly trained model
        Output      : Returns evaluation metrics and whether the new model is accepted.
        """
        logging.info("Entered the evaluate_model method of Model Evaluation class")
        try:
            # Reading the test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            
             # Define target column by summing the required columns
            logging.info("Since in this dataset target column was not defined , so we have defined it and later we do segregation")
            test_df["Shipment Price"] = (test_df["Line Item Value"] + test_df["Freight Cost (USD)"] + test_df["Line Item Insurance (USD)"])
            
            
            logging.info("Segregating the test_df data as independent and dependent columns")
            
            
            # Splitting into features and target
            logging.info("Droppping the Target Column")
            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]
            logging.info("Test data successfully split into features and target.")
            logging.info(f"Features (X) - Sample Data:\n{x.head().to_string(index=False)}")
            logging.info(f"Target (y) - Sample Data:\n{y.head().to_string(index=False)}")

            

            # Loading the trained model and predicting
            logging.info(f"trained_model_file_path : {self.model_trainer_artifact.trained_model_file_path}")
            trained_model = self.model_evaluation_config.UTILS.load_object(self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"trained_model is : {trained_model}")
            
            y_hat_trained_model = trained_model.predict(x)
            logging.info("Prediction done with the trained model")

            # Checking the R² score of the trained model
            trained_model_metrics = self.model_evaluation_config.UTILS.get_model_score(y, y_hat_trained_model)
            # Logging the metrics in a dynamic manner
            logging.info("The Metrics are Stored in Dictionary Form:")
            for metric_name, metric_value in trained_model_metrics.items():
                logging.info(f"{metric_name}: {metric_value}")

            
            trained_model_r2_score = trained_model_metrics["R2 Score"]
            
            # Evaluate the S3 model if it exists
            s3_model_r2_score = None
            s3_model = self.get_s3_model()
            if s3_model is not None:
                y_hat_s3_model = s3_model.predict(x)
                s3_model_metrics = self.model_evaluation_config.UTILS.get_model_score(y, y_hat_s3_model)
                # Logging the metrics in a dynamic manner
                logging.info("The S3 metrics are Stored in Dictionary Form:")
                for s3_metric_name, s3_metric_value in s3_model_metrics.items():
                    logging.info(f"{s3_metric_name}: {s3_metric_value}")
                    

                s3_model_r2_score = s3_model_metrics["R2 Score"]

            # Default to 0 if the S3 model is not available
            tmp_best_model_score = 0 if s3_model_r2_score is None else s3_model_r2_score

            # Logging the results
            is_model_accepted = trained_model_r2_score > tmp_best_model_score
            difference = trained_model_r2_score - tmp_best_model_score

            logging.info(f"Trained model R² score: {trained_model_r2_score}")
            logging.info(f"S3 model R² score: {s3_model_r2_score}")
            logging.info(f"Model difference: {difference}")

            logging.info("Exited the evaluate_model method of Model Evaluation Class")
            
            # Return the evaluation results
            return Evaluate_Model_Response(
                trained_model_r2_score=trained_model_r2_score,
                s3_model_r2_score=s3_model_r2_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

        except Exception as e:
            logging.info(f"Error occurred during model evaluation: {str(e)}")
            raise CustomException(str(e), sys)

    def initiate_model_evaluation(self) -> Model_Evaluation_Artifacts:
        """ 
        Method Name : initiate_model_evaluation
        Description : Initiates the model evaluation process and returns the evaluation artifact.
        Output      : Model Evaluation artifacts containing model performance details.
        """
        logging.info("Entered the initiate_model_evaluation method of Model_Evaluation class")
        try:
            # Initiating model evaluation
            evaluate_model_response = self.evaluate_model()

            # Saving model evaluation artifact
            model_evaluation_artifact = Model_Evaluation_Artifacts(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info("Exited the initiate_model_evaluation method of Model_Evaluation class")
            return model_evaluation_artifact

        except Exception as e:
            logging.info(f"Error occurred while initiating model evaluation: {str(e)}")
            raise CustomException(str(e), sys)
