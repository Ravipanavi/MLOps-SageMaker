from vehicle_insurance_prediction.configuration.mongo_db_connection import MongoDBClient
from vehicle_insurance_prediction.constants import DATABASE_NAME
from vehicle_insurance_prediction.exception import MyException
import pandas as pd
import sys
import numpy as np
import logging

class Proj1Data:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: str = None) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]
            
            # Debugging: Log document count
            doc_count = collection.count_documents({})
            logging.info(f"Found {doc_count} documents in collection '{collection_name}' of database '{self.mongo_client.database_name if database_name is None else database_name}'.")
            
            if doc_count == 0:
                raise ValueError(f"Collection '{collection_name}' is empty. Cannot export dataframe.")
            
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            
            df.replace({"na": np.nan}, inplace=True)
            return df
            
        except Exception as e:
            raise MyException(e, sys)

    def load_data(self):
        # TODO: implement data loading logic
        pass
