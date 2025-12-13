# src/vehicle_insurance_prediction/constants/training_pipeline.py
import os

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA_DIR = os.path.join(PACKAGE_ROOT, "schema")
SCHEMA_FILE_NAME = "schema.yaml"   # change if your file name differs
SCHEMA_FILE_PATH = os.path.join(SCHEMA_DIR, SCHEMA_FILE_NAME)

# other constants (examples)
ARTIFACT_DIR = os.path.join(PACKAGE_ROOT, "artifact")
TRAINING_PIPELINE_NAME = "vehicle_insurance_pipeline"
