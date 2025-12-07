import os
import sys
import pymongo
from dotenv import load_dotenv
from src.exception import MyException
from src.logger import logging

# Load .env FIRST before any other imports
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'), override=True)

class MongoDBClient:
    """MongoDB client connection wrapper."""
    
    def __init__(self, database_name: str = None):
        try:
            # Get fresh URI from environment (loaded from .env above)
            mongo_url = os.getenv("MONGODB_URL")
            
            if not mongo_url:
                raise ValueError("MONGODB_URL environment variable not found in .env!")
            
            # Debug: log the URI (mask password)
            masked_uri = mongo_url.split('@')[0] + '@***' if '@' in mongo_url else mongo_url[:20] + '...'
            logging.info(f"Connecting with URI: {masked_uri}")
            
            self.mongo_url = mongo_url
            self.database_name = database_name
            
            # Create connection
            self.client = pymongo.MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            logging.info(f"MongoDB connection successful to {database_name}")
            
            # Access database
            self.database = self.client[database_name]
            
        except Exception as e:
            logging.error(f"MongoDB connection failed: {e}")
            raise MyException(e, sys)
    
    def __getitem__(self, key):
        """Allow accessing collections."""
        return self.database[key]
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed")