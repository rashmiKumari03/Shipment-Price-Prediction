from src.Shipment_Price_Prediction.logger import logging
from src.Shipment_Price_Prediction.exception import CustomException
import sys


# Example to cross check wheather logger working or not?
logging.info("Crosschecking wheather logger is working or not?")
logging.info("Hello Learners!!!")
print("Lets Learn Data Science.")   
logging.info("Yeah! My logger is working fine.")

# In Terminal : Run   " python trail_app.py "
# Observation : We have to use logging.info on the things we want to log.

# Example to cross check wheather exception working or not?
logging.info("Crosschecking wheather exception working or not?")

try:
    
    num = 12000 / 0
    print(num)
    
except Exception as e:
    # Logging the Issue 
    logging.info(CustomException(e,sys))
    raise CustomException(e,sys)
