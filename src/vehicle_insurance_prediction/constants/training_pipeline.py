# src/vehicle_insurance_prediction/constants/training_pipeline.py

import os

# Artifact directories
ARTIFACT_DIR = "artifact"
MODEL_DIR = os.path.join(ARTIFACT_DIR, "model")
DATA_DIR = os.path.join(ARTIFACT_DIR, "data")

# Schema file (relative to project)
SCHEMA_DIR = "config"
SCHEMA_FILE_NAME = "schema.yaml"
SCHEMA_FILE_PATH = os.path.join(SCHEMA_DIR, SCHEMA_FILE_NAME)

# Target and features (change TARGET_COLUMN to your real column name)
TARGET_COLUMN = "Response"
NUMERICAL_COLUMNS = [
    "Age",
    "Region_Code",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]
CATEGORICAL_COLUMNS = [
    "Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage"
]

# Training settings
CURRENT_YEAR = 2025
RANDOM_STATE = 42
TEST_SIZE = 0.2
PIPELINE_NAME = "vehicle_insurance"

# Data Ingestion related constants
DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DATABASE_NAME: str = "proj1"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model Trainer related constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_N_ESTIMATORS: int = 200
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MIN_SAMPLES_SPLIT_MAX_DEPTH: int = 10
MIN_SAMPLES_SPLIT_CRITERION: str = 'entropy'
MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 42

# Model Evaluation/Pusher related constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "rrr-mlops-sagemaker"

# Common file names
FILE_NAME: str = "vehicle_insurance.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
