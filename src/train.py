import os
import argparse
import shutil
import joblib
import pandas as pd
from vehicle_insurance_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from vehicle_insurance_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from vehicle_insurance_prediction.components.data_ingestion import DataIngestion
from vehicle_insurance_prediction.components.data_validation import DataValidation
from vehicle_insurance_prediction.components.data_transformation import DataTransformation
from vehicle_insurance_prediction.components.model_trainer import ModelTrainer
from vehicle_insurance_prediction.exception import VehicleException
from vehicle_insurance_prediction.logger import logging


def start_training(output_path, model_dir):
    """
    This function orchestrates the MLOps pipeline for training the model.
    It's adapted to run within a SageMaker training job.
    """
    try:
        # Define configurations. These might need adjustments for SageMaker.
        # For example, paths might be relative to SageMaker's environment.
        training_pipeline_config = {
            "pipeline_name": "sagemaker-training-pipeline",
            "artifact_dir": os.path.join(output_path, "artifact"),
        }
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )

        # 1. Data Ingestion
        logging.info("Starting data ingestion.")
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")

        # 2. Data Validation
        logging.info("Starting data validation.")
        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed.")

        # 3. Data Transformation
        logging.info("Starting data transformation.")
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed.")

        # 4. Model Trainer
        logging.info("Starting model training.")
        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config
        )
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config,
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed.")

        # SageMaker requires the model to be saved in a specific directory.
        # We copy the trained model to the path specified by SageMaker.
        logging.info(f"Copying trained model to {model_dir}")
        trained_model_path = model_trainer_artifact.trained_model_file_path
        
        # To be compatible with the inference.py script, let's rename it to model.pkl
        destination_path = os.path.join(model_dir, "model.pkl")
        shutil.copy(trained_model_path, destination_path)
        
        logging.info(f"Model saved at {destination_path}")

    except Exception as e:
        logging.error(f"Training pipeline failed with exception: {e}")
        raise VehicleException(e) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments.
    # See: https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#prepare-a-scikit-learn-training-script
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args = parser.parse_args()

    # The MONGODB_URL should be set as an environment variable in the SageMaker training job configuration.
    logging.info("SageMaker training job started.")
    start_training(output_path=args.output_data_dir, model_dir=args.model_dir)
    logging.info("SageMaker training job completed successfully.")