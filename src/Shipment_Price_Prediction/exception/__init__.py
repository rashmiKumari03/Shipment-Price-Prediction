import sys
from src.Shipment_Price_Prediction.logger import logging

def error_message_detail(error, error_detail: sys):
    # Try to get the traceback details, handle None in case of missing traceback
    try:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    except Exception as e:
        # If traceback is not available, provide a simplified error message
        error_message = f"Error occurred with message [{str(error)}], but no traceback info available."
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Inherit from Exception
        
        # Get detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# Now after this go to  trail_app.py and call it there and check whether it is working fine of not.