import os
from os import environ

DB_URL = environ["MONGO_DB_URL"]  # We need to set the mongodb url to our local environment Variable..
                                  # By using MONGO_DB_URL it will read the url
                    
                     
