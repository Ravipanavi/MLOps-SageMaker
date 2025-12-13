import os
from vehicle_insurance_prediction.constants.training_pipeline import *
from vehicle_insurance_prediction.constants.s3_bucket import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name: str = PIPELINE_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        self.timestamp: str = TIMESTAMP


@dataclass
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.validation_report_file_path: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                        TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                       TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir,
                                                         DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                         PREPROCESSING_OBJECT_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
        self._n_estimators = MODEL_TRAINER_N_ESTIMATORS
        self._min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
        self._min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
        self._max_depth = MIN_SAMPLES_SPLIT_MAX_DEPTH
        self._criterion = MIN_SAMPLES_SPLIT_CRITERION
        self._random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.bucket_name: str = MODEL_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bucket_name: str = MODEL_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class VehiclePredictorConfig:
    def __init__(self):
        self.model_file_path: str = MODEL_FILE_NAME
        self.model_bucket_name: str = MODEL_BUCKET_NAME