import os
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO)

project_name = 'Shipment_Price_Prediction'


list_of_files=[
    
    
    f"src/{project_name}/__init__.py",
    
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    f"src/{project_name}/components/model_predictor.py",
    
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/mongo_operation.py",
    f"src/{project_name}/configuration/s3_operation.py",
    
    f"src/{project_name}/constant/__init__.py",
    
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifacts_entity.py",


    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    
    
    
    "trail_app.py",
    "app.py",
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "setup.py",
    "config/model.yaml", 
    "config/schema.yaml",
    
    
    "templates/index.html",
    "static/css/style.css"
]

# Now code will execute these above paths and make directories.

for filepath in list_of_files:
    filepath = Path(filepath)  # We are converting the string to actual path using Path
    filedir , filename = os.path.split(filepath)  

    # If file dir is not equal to "" ie. if file dir is non empty...means there is something in filedir then make the dir with that address/path.

    if filedir != "":
        os.makedirs(filedir,exist_ok=True) 
        logging.info(f"Creating directory:{filedir} for the file {filename}")
    
    # if that above path doesnot exist or path size is zero ==> path doesnot exist...then we will create an empty filepath..else filename already exist.
    # Suppose we already have setup.py and requirements.txt whose size is non zero therefore it will skip the if part and comes to else and prints file already exist.
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file:{filepath}")
    else:
        logging.info(f"{filename} is already exists")


# Run it in terminal : python template.py


    """
    Project Structure Overview
    
    Source Directory (src/{project_name})
    
    __init__.py :  Purpose: Marks the directory as a Python package, allowing for modular imports.
    
    Components
            components/__init__.py : Initializes the components package.
            data_ingestion.py : Handles the ingestion of data from various sources (e.g., databases, files).
            data_validation.py : Validates the ingested data against predefined schemas or rules.
            data_transformation.py : Transforms the data into a suitable format for modeling (e.g., normalization, encoding).
            model_trainer.py : Contains logic for training machine learning models using the prepared data.
            model_evaluation.py : Evaluates the performance of trained models using metrics like accuracy, precision, etc.
            model_pusher.py : Manages the deployment of trained models to production environments.
            model_predictor.py : Provides functionality for making predictions with deployed models.
            
    Pipelines
            pipelines/__init__.py : Initializes the pipelines package.
            training_pipeline.py : Orchestrates the entire training process, integrating data ingestion, validation, transformation, and model training.
            prediction_pipeline.py : Manages the prediction workflow, including data input and model inference.
                        
    Utilities
            utils/__init__.py : Initializes the utilities package 
            main_utils.py : Contains utility functions that can be reused across different components (e.g., logging, configuration loading).
            
    Configuration
            configuration/__init__.py : Initializes the configuration package.
            mongo_operation.py : Handles operations related to MongoDB (e.g., data retrieval and storage).
            s3_operation.py : Manages interactions with Amazon S3 for data storage and retrieval.
            
    Constants
            constant/__init__.py : Initializes the constants package.
            This directory typically contains constant values used throughout the project (e.g., file paths, model names).
            
    Entities
            entity/__init__.py : Initializes the entity package.
            config_entity.py : Defines configuration entities that encapsulate various configuration settings (e.g., model parameters).
            artifacts_entity.py : Defines entities related to artifacts generated during model training and evaluation (e.g., model files, metrics).
            
    Exception Handling
            exception/__init__.py : Initializes the exception handling package.
            This directory typically contains custom exception classes for better error handling.
            
    Logging
            logger/__init__.py : Initializes the logging package.
            This directory usually includes configurations and functions for logging application events.
            
    Root Directory Files
            trail_app.py : A script that may serve as an entry point or main application logic for running experiments or testing.
            app.py : Another entry point that might be used to start a web application or API service.
            Dockerfile : Contains instructions for building a Docker image of your application, ensuring consistent environments across different deployments.
            .dockerignore : Specifies files and directories that should be ignored when building a Docker image (similar to .gitignore, but for Docker).
            requirements.txt : Lists all Python dependencies required to run your project. It is used by pip to install necessary packages.
            setup.py : A script for packaging your project as a Python package. It defines metadata about your project and its dependencies.
            Configuration Files
                config/model.yaml  : Stores model parameters, hyperparameters, and settings necessary for training and evaluating machine learning models.
                config/schema.yaml : Defines data types and schema configurations used for validating input data before processing it through the pipeline.
    
    Frontend Files
            templates/index.html : An HTML file that serves as a template for rendering web pages in a web application.
            static/css/style.css : Contains CSS stylesheets used to style HTML templates in your web application.
    """

