from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import sys
from src.Shipment_Price_Prediction.pipelines.training_pipeline import TrainPipeline


if __name__=='__main__':
    obj = TrainPipeline()
    obj.run_pipeline()