import sys
from src.exception import MyException
from src.logger import logging

class MyModel:
    """
    A custom model class that encapsulates a preprocessing object and a trained model object
    to streamline the prediction process.
    """
    def __init__(self, preprocessing_object, trained_model_object):
        """
        Initializes the MyModel instance.
        :param preprocessing_object: A scikit-learn pipeline or transformer for preprocessing input data.
        :param trained_model_object: A trained machine learning model object.
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        Generates predictions for the given input data.
        :param X: Input data (e.g., a pandas DataFrame) to be used for prediction.
        :return: The predictions made by the model.
        """
        try:
            logging.info("Transforming input features and making predictions.")
            transformed_feature = self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise MyException(e, sys) from e