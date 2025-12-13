# src/train.py
import os
import argparse
import shutil
import logging
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VehicleException(Exception):
    pass

def start_training(output_path: str, model_dir: str):
    try:
        logging.info("Starting training pipeline (imports and configs)...")
        try:
            from vehicle_insurance_prediction.entity.config_entity import (
                DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
                ModelTrainerConfig, TrainingPipelineConfig
            )
            from vehicle_insurance_prediction.components.data_ingestion import DataIngestion
            from vehicle_insurance_prediction.components.data_validation import DataValidation
            from vehicle_insurance_prediction.components.data_transformation import DataTransformation
            from vehicle_insurance_prediction.components.model_trainer import ModelTrainer
            from vehicle_insurance_prediction.constants.training_pipeline import SCHEMA_FILE_PATH
        except Exception as imp_ex:
            logging.error("Failed to import project modules. Check that the package was built & installed correctly.")
            logging.error("Import exception: %s", imp_ex)
            logging.error(traceback.format_exc())
            raise

        if not model_dir:
            model_dir = "/opt/ml/model"
            logging.warning("model_dir not provided; defaulting to %s", model_dir)
        if not output_path:
            output_path = "/opt/ml/output/data"
            logging.warning("output_path not provided; defaulting to %s", output_path)

        training_pipeline_config = TrainingPipelineConfig()

        data_ingestion = DataIngestion(DataIngestionConfig(training_pipeline_config))
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")

        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=DataValidationConfig(training_pipeline_config),
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed.")

        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=DataTransformationConfig(training_pipeline_config),
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed.")

        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=ModelTrainerConfig(training_pipeline_config),
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed.")

        trained_model_path = getattr(model_trainer_artifact, "trained_model_file_path", None)
        if not trained_model_path or not os.path.exists(trained_model_path):
            raise VehicleException(f"Trained model file not found at {trained_model_path}")

        os.makedirs(model_dir, exist_ok=True)
        destination_path = os.path.join(model_dir, "model.pkl")
        shutil.copy(trained_model_path, destination_path)
        logging.info("Saved trained model to %s", destination_path)

        try:
            schema_dest = os.path.join(output_path, "schema_path.txt")
            with open(schema_dest, "w") as fh:
                fh.write(str(SCHEMA_FILE_PATH))
            logging.info("Wrote schema path to %s", schema_dest)
        except Exception:
            logging.warning("Could not write schema path to output; continuing.")

        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error("Training pipeline failed: %s", e)
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    args = parser.parse_args()
    logging.info("SageMaker training job started.")
    start_training(output_path=args.output_data_dir, model_dir=args.model_dir)
    logging.info("SageMaker training job completed (script exit).")
