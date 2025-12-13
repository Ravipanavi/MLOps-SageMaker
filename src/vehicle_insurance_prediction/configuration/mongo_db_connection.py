import pymongo
import os
import sys
from vehicle_insurance_prediction.constants import DATABASE_NAME, MONGODB_URL_KEY
from vehicle_insurance_prediction.logger import logging
from vehicle_insurance_prediction.exception import MyException

class MongoDBClient:
    """
    Class to handle MongoDB connections.
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                
                # Masking for logs
                if "@" in mongo_db_url:
                    masked_url = mongo_db_url.split("@")[0] + "@***"
                else:
                    masked_url = mongo_db_url[:20] + "..."
                logging.info(f"Connecting to MongoDB: {masked_url}")

                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, serverSelectionTimeoutMS=5000)
                MongoDBClient.client.admin.command('ping') # Verify connection
                logging.info("MongoDB connection successful (ping verified).")
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise MyException(e, sys)