import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Tuple
from from_root.root import from_root


from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
from src.Shipment_Price_Prediction.utils.main_utils import MainUtils
from src.Shipment_Price_Prediction.constant import MODEL_CONFIG_FILE, SCHEMA_FILE_PATH 
from src.Shipment_Price_Prediction.components.data_transformation import Data_Transformation
from src.Shipment_Price_Prediction.entity.config_entity import Model_Trainer_Config
from src.Shipment_Price_Prediction.entity.artifacts_entity import Data_Transformation_Artifacts, Model_Trainer_Artifacts

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


class Model_Trainer:

    def __init__(self, 
                 data_transformation_artifact: Data_Transformation_Artifacts, 
                 model_trainer_config: Model_Trainer_Config):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        
         
        UTILS = MainUtils()
        self.SCHEMA_CONFIG = UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
        self.numeric_columns = self.SCHEMA_CONFIG.get('numeric_features',[])
        
        
    
    # This method is used to get the trained model.
    def get_trained_models(self, train_data: DataFrame, test_data: DataFrame) -> List[Tuple[float, object, str]]:
        """
        
        Method Name : get_trained_models
        
        Description : Retrieve trained models from config and train them.
        
        Output      : List of trained models
        """
        logging.info("Entered get_trained_models method of Model_Trainer class.")
        try:
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            models_list = list(model_config.get("train_model", {}).keys())
            logging.info(f"Got the models list from config file to train: {models_list}")
            
            '''
            Usually, splitting occurs when both the independent and dependent data are in the same training set. 
            However, in this case, since my independent and dependent data are already saved in separate files in .npz format, 
            I will load them directly."
            
            '''
            
            logging.info(f"Training data: {train_data.shape}, Test data: {test_data.shape}")
            
            X_train , y_train ,   X_test , y_test  = (train_data.drop(train_data.columns[len(train_data.columns)-1],axis=1) ,
                                                      train_data.iloc[:,-1] ,
                                                      test_data.drop(test_data.columns[len(test_data.columns)-1],axis=1) , 
                                                      test_data.iloc[:,-1])         
            
          
            
            trained_models = [
                self.model_trainer_config.UTILS.get_tunned_model(model_name,X_train, y_train,X_test, y_test)
                for model_name in models_list
            ]

            logging.info("Model training completed.")
            logging.info("Got Trained Model List")
            logging.info("Exited the get_trained_models method of Model_Trainer class")
            
            return trained_models

        except Exception as e:
            logging.error(f"Error in get_trained_models: {str(e)}")
            raise CustomException(str(e), sys)


    # This method is used to initialize model training
    def initiate_model_trainer(self) -> Model_Trainer_Artifacts:
        """
        Method Name : initiate_model_trainer
        
        Description : Initiate model training, select the best model, and save it.
        
        Output      : List of trained models
        """
        logging.info("Entered initiate_model_trainer method of Model_Trainer class.")
        try:
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            
            train_arr = pd.DataFrame(self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.train_file_path))
            test_arr = pd.DataFrame(self.model_trainer_config.UTILS.load_numpy_array_data(self.data_transformation_artifact.test_file_path))
            
            logging.info(f"X_train looks like : {train_arr.head()}")
            logging.info(f"X_test looks like : {test_arr.head()}")
            logging.info("===========================================")

            trained_models = self.get_trained_models(train_arr, test_arr)
            
            best_model_name,best_model_object, best_model_score = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(trained_models)
            
            model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
            base_model_score = float(model_config["base_model_score"])

            if best_model_score >= base_model_score:
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                
                preprocessor_obj = self.model_trainer_config.UTILS.load_object(self.data_transformation_artifact.transformed_object_file_path)
                logging.info("Here i am Saving the cost_model using save_object!!!!")
                cost_model = Cost_Model(preprocessor_obj, best_model_object)
                logging.info("Created a Cost_Model with preprocessor and model")
                logging.info(f"cost_model looks like : {cost_model} and type of cost_model is {type(cost_model)}")
                
                
                model_file_path = self.model_trainer_config.UTILS.save_object(self.model_trainer_config.TRAINED_MODEL_FILE_PATH, cost_model)
                logging.info(f"model_file_path looks like : {model_file_path}")
                model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=model_file_path)
                logging.info(f"Model saved at {model_file_path}")
                logging.info(f"Model Trainer Artifacts : {model_trainer_artifacts}")

                return model_trainer_artifacts
            
            else:
                artifacts_path = os.path.join(from_root(), "artifacts")
                logging.info("Fetching list of subfolders inside the artifact directory.")

                # Get all folder names sorted by creation time
                dir_list = sorted(
                    (folder_name for folder_name in os.listdir(artifacts_path) 
                    if os.path.isdir(os.path.join(artifacts_path, folder_name))),
                    key=lambda x: os.path.getmtime(os.path.join(artifacts_path, x))
                )

                # Log the folder names
                for folder_name in dir_list:
                    logging.info(f"Found folder: {folder_name}")

                # Return the second-to-last folder name if it exists
                if len(dir_list) >= 2:
                    second_last_folder = dir_list[-2]
                    logging.info(f"Loading model from the second-to-last artifact directory: {second_last_folder}")
                else:
                    logging.warning("Fewer than two artifact folders found. Exiting the method.")
                    return None  
                
                # Define the path to the Model_Trainer_Artifacts folder
                model_artifacts_path = os.path.join(artifacts_path, second_last_folder, 'Model_Trainer_Artifacts')
                 
                # List all files inside Model_Trainer_Artifacts
                files_in_artifacts = os.listdir(model_artifacts_path)
                logging.info(f"Files inside Model_Trainer_Artifacts:\n {files_in_artifacts}")
                
                # Convert list to string
                files_in_artifacts = ''.join(files_in_artifacts )
                                
                # Read the required file (assuming you want to load a specific file, e.g., MODEL_FILE_NAME)
                model_file_path = os.path.join(model_artifacts_path, files_in_artifacts)

                # Error handling for the model file loading
                try:
                    base_model_object = self.model_trainer_config.UTILS.load_object(model_file_path)
                except Exception as e:
                    logging.error(f"Error loading model from {model_file_path}: {e}")
                    return None  # Handle as appropriate, e.g., raise an exception

                # Save the base model directly without additional wrapping
                output_model_file_path = os.path.join(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, "Retained_Model.pkl")
                self.model_trainer_config.UTILS.save_object(output_model_file_path, base_model_object)
                logging.info(f"Base model re-saved at {output_model_file_path}.")

                # Create and return the model trainer artifacts
                model_trainer_artifacts = Model_Trainer_Artifacts(trained_model_file_path=output_model_file_path)
                return model_trainer_artifacts
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(str(e), sys)




"""
Note : MODEL SELECTION, SAVING, AND S3 PUSH LOGIC

We follow a two-stage strategy to evaluate, name, and save machine learning models
based on their R2 performance. This ensures only the best-performing model is retained
locally and optionally pushed to S3 for production use.

===============================================================================
STAGE 1: LOCAL COMPARISON — best_trained_model vs base_model

Why:
To check whether the newly trained model performs better than the baseline (base model).

When:
Immediately after local training is completed.

How:
- Evaluate both models on the same unseen test dataset (X_test) using R2 score.

Condition:
if best_trained_model_r2_score >= base_model_r2_score:
    → Accept the trained model.
    → Save it as "Shipment_Price_Model.pkl" (this becomes our selected local model).
else:
    → Reject the trained model.
    → Retain the base model.
    → Save it as "Retained_Model.pkl" (this becomes our selected local model).

Important:
- Only one model moves forward to the next stage — the better one.
- "Retained_Model.pkl" is only created here if base model wins.

===============================================================================
STAGE 2: REMOTE COMPARISON — selected local model vs S3 model

Why:
To decide whether the selected local model (from Stage 1) is better than
the current production model stored in S3.

When:
Before pushing any model to S3.

How:
- Load the existing model from S3 and evaluate its R2 score (s3_model_r2_score).
- Handle cases where no S3 model exists by using:
      tmp_best_model_score = 0 if s3_model_r2_score is None else s3_model_r2_score

Condition:
if trained_model_r2_score > tmp_best_model_score:
    → Push the selected local model to S3.
    → Save it in S3 using the filename "Shipment_Price_Model.pkl".
else:
    → Do not push to S3.
    → No new file is saved locally at this stage.
    → "Retained_Model.pkl" is not created here again.

Important:
- Whether the selected local model was trained or retained, it is always
  pushed to S3 with the name "Shipment_Price_Model.pkl" if it performs better.

===============================================================================
NAMING RULES SUMMARY

✔ "Shipment_Price_Model.pkl":
    - Used when a model (trained or retained) is better than the base.
    - Also always used for the model pushed to S3 for consistent naming.

✔ "Retained_Model.pkl":
    - Used only in Stage 1 when base_model is better than trained model.
    - Never created again in Stage 2.

===============================================================================

"""
