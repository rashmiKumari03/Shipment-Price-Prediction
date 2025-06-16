import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException


from src.Shipment_Price_Prediction.constant import *
from src.Shipment_Price_Prediction.entity.artifacts_entity import (Data_Ingestion_Artifacts, Data_Transformation_Artifacts,Model_Trainer_Artifacts, Model_Evaluation_Artifacts)
from src.Shipment_Price_Prediction.entity.config_entity import Model_Evaluation_Config

from src.Shipment_Price_Prediction.components.model_trainer import Cost_Model

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
   
   
   
Model Evaluation Class: The Model_Evaluation class is responsible for evaluating the performance of a newly trained model against a previously deployed model (stored in S3).

Key Methods:

- get_s3_model: Fetches the model from the S3 bucket if it exists.
- evaluate_model: Compares the newly trained model with the S3 model by calculating their R² scores and determining if the new model is better.
- initiate_model_evaluation: Initiates the evaluation process and returns the evaluation artifacts.

"""

# This class elements were logged.
@dataclass
class Evaluate_Model_Response:
    trained_model_r2_score: float
    s3_model_r2_score: float
    is_model_accepted: bool
    difference: float


class Model_Evaluation:
    
    def __init__(self, model_trainer_artifact: Model_Trainer_Artifacts,
                 model_evaluation_config: Model_Evaluation_Config, 
                 data_ingestion_artifact : Data_Ingestion_Artifacts,
                 data_transformation_artifact: Data_Transformation_Artifacts):
        
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_artifact = data_transformation_artifact
        
        
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
                logging.info(f"S3 model loaded: {model}")
                logging.info(f"S3 model object id: {id(model)}")
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
            logging.info("Readind the Test dataset from Ingestion")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            logging.info(f"Test_df : {test_df.head()}")
        
            
            # Splitting into input features and target features
            X_test = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            Y_test = test_df[TARGET_COLUMN]
            logging.info(f"X_test Looks like : {X_test.head()}")
            logging.info(f"Y_test Looks like : {Y_test.head()}")
            logging.info(f"X_test and Y_test has been loaded!")
            
            
    
            # Loading the trained model and prediction
            logging.info(f"trained_model_file_path : {self.model_trainer_artifact.trained_model_file_path}")

            # Use the load_object method to load the trained model, this is the cost_model which was stored in the trainer part.
            trained_model = self.model_evaluation_config.UTILS.load_object(self.model_trainer_artifact.trained_model_file_path)

            logging.info(f"trained_model is : {trained_model}")
            
            logging.info("Prediction using preprocessor_obj and trained_model initiated!")
            
            Y_test = np.log1p(Y_test)
            logging.info(f"Y_test looks like : {Y_test}")

            # Use the instance to call predict
            Y_hat_trained_model = trained_model.prediction(X_test)
            logging.info(f"Y_hat_trained_model looks like : {Y_hat_trained_model}")
            
            logging.info("Prediction done with the trained model")
            
            
            # Convert predictions and actual values to DataFrames for consistency
            Y_test_df = pd.DataFrame(Y_test, columns=['Actual'])
            Y_hat_trained_model_df = pd.DataFrame(Y_hat_trained_model, columns=['Predicted'])

            # Log the predictions
            logging.info(f"Y_test:\n{Y_test_df.head()}")
            logging.info(f"Y_hat_trained_model:\n{Y_hat_trained_model_df.head()}")

    
            # Checking the R² score of the trained model
            trained_model_metrics = self.model_evaluation_config.UTILS.get_model_score(Y_test, Y_hat_trained_model)
            # Logging the metrics in a dynamic manner
            logging.info("The Metrics are Stored in Dictionary Form:")
            for metric_name, metric_value in trained_model_metrics.items():
                logging.info(f"{metric_name}: {metric_value}")

            
            trained_model_r2_score = trained_model_metrics["R2 Score"]
        
            
            
            # ------------------------------------------------------------------------------------------------------------------
            # Evaluate the S3 model if it exists
            s3_model_r2_score = None
            s3_model = self.get_s3_model()
            if s3_model is not None:
                
                logging.info("S3 bucket is not empty — a model exists!")
                logging.info("Initiating prediction using the S3-trained model.")

                Y_hat_s3_model = s3_model.prediction(X_test)
                s3_model_metrics = self.model_evaluation_config.UTILS.get_model_score(Y_test, Y_hat_s3_model)
                # Logging the metrics in a dynamic manner
                logging.info("The S3 metrics are Stored in Dictionary Form:")
                for s3_metric_name, s3_metric_value in s3_model_metrics.items():
                    logging.info(f"{s3_metric_name}: {s3_metric_value}")
                    

                s3_model_r2_score = s3_model_metrics["R2 Score"]
                

            logging.info("Now will do the comparison")

            # Default to 0 if the S3 model is not available
            tmp_best_model_score = 0 if s3_model_r2_score is None else s3_model_r2_score

            # Logging the results
            # trained_model_r2_score = 0.99999998  # Manually set a high R2 score to test if the code correctly pushes better models to S3

            is_model_accepted = trained_model_r2_score > tmp_best_model_score
            difference = trained_model_r2_score - tmp_best_model_score
            
            """
            Note:
            ---------
            - Always compare both R2 score and model ID
            - Same R2 does not guarantee same model
            """     
            # Step 1: Log the R2 score and ID of the newly trained model
            logging.info(f"[Step 1] Newly trained model evaluated on unseen test data (X_test).")
            logging.info(f"Trained model R2 score: {trained_model_r2_score}")
            logging.info(f"Trained model ID (memory reference): {id(trained_model_r2_score)}")

            # Step 2: Log the R2 score and ID of the current best model stored in S3
            logging.info(f"[Step 2] Comparing with existing best model retrieved from S3.")
            logging.info(f"Present S3 model R2 score: {tmp_best_model_score}")
            logging.info(f"S3 model ID (memory reference): {id(tmp_best_model_score)}")

            # Step 3: Log whether the models have the same R2 score
            logging.info(f"[Step 3] Are both models performing equally?")
            logging.info(f"Are both R2 scores same? {trained_model_r2_score == tmp_best_model_score}")

            # Step 4: Log the actual difference in R2 scores
            difference = trained_model_r2_score - tmp_best_model_score
            logging.info(f"[Step 4] Performance difference between new and S3 model:")
            logging.info(f"R2 score difference (new - existing): {difference:.6f}")

            # Step 5: Log the decision based on R2 score comparison
            if trained_model_r2_score > tmp_best_model_score:
                logging.info(f"New model accepted: It outperforms the existing model.")
                logging.info(f"Action: Saving and uploading the new model to S3.")
            else:
                logging.info(f"New model rejected: It does not improve over the existing model.")
                logging.info(f"Action: Keeping the existing S3 model unchanged.")

        


            logging.info("Exited the evaluate_model method of Model Evaluation Class")
            
            # Return the evaluation results
            return Evaluate_Model_Response(
                trained_model_r2_score=trained_model_r2_score,
                s3_model_r2_score=s3_model_r2_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

        except Exception as e:
            logging.info(CustomException(str(e),sys))
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
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e), sys)



"""
---------------------- Model Evaluation FAQ (Clarification Note) ----------------------

Q1: Why do we use `random_state` in `train_test_split`?
A1: Setting `random_state=42` ensures that the data is split into training and test sets the same way every time. 
    This guarantees that the evaluation is always done on a consistent `X_test` and `y_test`, allowing for fair 
    comparison between models.

Q2: Does using the same `random_state` ensure identical R² scores for trained and S3 models?
A2: No, it only ensures the test dataset (`X_test`) remains the same. R² scores can still differ because:
    - Training might involve internal randomness (e.g., XGBoost's boosting process).
    - Models might have been trained in different sessions or with different hyperparameters.
    - Even the same model architecture (e.g., XGBoost) can produce slightly different results if training conditions 
      or pipelines differ.

Q3: Why can the S3 model's R² score differ from what it was at the time of deployment?
A3: The S3 model is re-evaluated on the current `X_test` at runtime. If anything (like feature processing or data 
    distribution) has changed slightly, or if randomness was involved during training, the new R² can differ slightly 
    from the original logged score.

Q4: When is the newly trained model considered "better" than the S3 model?
A4: If the newly trained model has a higher R² score than the S3 model when both are evaluated on the current `X_test`, 
    it is accepted as better. The difference in R² is logged and used to decide whether to push the new model.

----------------------------------------------------------------------------------------
"""


