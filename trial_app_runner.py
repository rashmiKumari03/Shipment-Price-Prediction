from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import sys
from src.Shipment_Price_Prediction.pipelines.training_pipeline import TrainPipeline
import warnings
warnings.filterwarnings("ignore")



if __name__=='__main__':
    logging.info("App Running....")
    obj = TrainPipeline()
    obj.run_pipeline()
    
    
# Firstly on running this in terminal : We will get the artifacts folder : having Train and Test data.