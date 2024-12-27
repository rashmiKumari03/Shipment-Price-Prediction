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
