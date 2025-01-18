from dataclasses import dataclass

# Data Ingestion Artifacts

@dataclass
class Data_Ingestion_Artifacts:
    """
    This class holds the file paths to the train and test data that are created after data ingestion.

    Attributes:
    - train_data_file_path (str): This is the path where the training data file is saved after ingestion.
    - test_data_file_path (str): This is the path where the testing data file is saved after ingestion.

    Purpose:
    After the data ingestion process, the train and test data are saved as files. 
    This class helps in storing and easily accessing the file paths of those saved data files.
    These paths are later used when training the machine learning model or testing its performance.
    """
    train_data_file_path: str
    test_data_file_path: str
    
    
    
# Data Validation Artifacts

@dataclass
class Data_Validation_Artifacts:
    """
    This class holds the information and file paths generated during the data validation process.

    Attributes:
    - data_drift_file_path (str): The path to the file that contains the data drift report. 
      This report indicates if there is any significant difference between the distribution of training and testing data or between new data and the original dataset.
    - validation_status (bool): A flag indicating whether the data validation process was successful or not. 
      If True, the data passed all validation checks (e.g., schema validation, missing value checks, etc.). 
      If False, issues were found during the validation process.

    Purpose:
    After the data validation process, artifacts such as the data drift report and validation status are generated. 
    This class helps in storing and accessing this information for further steps in the machine learning pipeline, 
    ensuring the data is reliable and ready for model training or evaluation.
    """
    data_drift_file_path: str
    validation_status: bool
    
    
# Data Transformation Artifacts

@dataclass
class Data_Transformation_Artifacts:
    """
    This class holds the file paths and information generated during the data transformation process.

    Attributes:
    - transformed_object_file_path (str): The path to the file that contains the transformation object (e.g., scaler, encoder) 
      used during data preprocessing. This object is used to apply consistent transformations to both training and testing data, 
      as well as to any new data during inference.
    - transformed_train_file_path (str): The path to the file containing the transformed training data. 
      This data has been preprocessed and is ready to be used for model training.
    - transformed_test_file_path (str): The path to the file containing the transformed testing data. 
      This data has been preprocessed and is ready to be used for evaluating the model.

    Purpose:
    The data transformation process ensures that the raw data is converted into a format suitable for machine learning models. 
    This includes operations like scaling, encoding categorical features, and handling missing values. 
    This class stores the paths to the artifacts generated during this process, which are essential for maintaining 
    consistency between training and inference.
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

    
    
@dataclass
class Model_Trainer_Artifacts:
    """
    This class holds the information and file paths generated during the model training process.

    Attributes:
    - trained_model_file_path (str): The path where the trained model is saved after the training process is completed.
      This model file contains the weights, and other necessary parameters that define the trained machine learning model.

    Purpose:
    After the model training process, the trained model is saved to a file. This class stores the path to that saved model,
    making it easily accessible for later stages such as evaluation, deployment, or inference. The saved model can be used to
    predict outcomes based on new, unseen data.
    """
    trained_model_file_path: str

  
@dataclass
class Model_Evaluation_Artifact:
    """
    This class holds the information related to the evaluation results of the trained model.

    Attributes:
    - is_model_accepted (bool): A boolean flag indicating whether the trained model meets the performance 
      criteria and is accepted for further use (e.g., deployment). If True, the model has passed evaluation 
      and is deemed ready for deployment. If False, the model did not meet the desired standards and further 
      improvements may be needed.
      
    - trained_model_path (str): The file path where the trained model is stored. This is important for tracking
      which model was evaluated and linking the evaluation results back to the correct version of the model.

    - changed_accuracy (float): The change in accuracy (or any other performance metric) after the evaluation. 
      This attribute is used to track the performance difference compared to a baseline (e.g., previous model version 
      or threshold). A positive value indicates an improvement in accuracy, while a negative value indicates a degradation.

    Purpose:
    After a model is trained, it needs to be evaluated to assess its performance. This class stores the results 
    of the model evaluation, including whether the model is accepted, the path to the trained model, 
    and the change in its accuracy. These results are crucial for deciding whether the model should proceed to 
    the deployment phase or if further training and refinement are required. The evaluation helps determine the model's 
    readiness to make predictions on unseen data and if it can provide value in real-world applications.
    """
    is_model_accepted: bool
    trained_model_path: str
    changed_accuracy: float
