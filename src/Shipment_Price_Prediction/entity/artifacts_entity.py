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
