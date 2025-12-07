import json
import os
import joblib
import pandas as pd


def model_fn(model_dir):
    """
    Loads the model from the model directory.

    Args:
        model_dir (str): The directory where the model artifacts are stored.
                         SageMaker will place the contents of your model.tar.gz here.

    Returns:
        object: The loaded model object.
    """
    print("Loading model from a .pkl file")
    # Assuming the model is saved as 'model.pkl' in your training pipeline
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body into a pandas DataFrame.

    Args:
        request_body (str): The body of the request.
        request_content_type (str): The content type of the request.

    Returns:
        pandas.DataFrame: The deserialized input data.
    """
    if request_content_type == "application/json":
        print("Deserializing JSON input.")
        input_data = json.loads(request_body)
        # The input is expected to be a dictionary or a list of dictionaries
        # that can be converted to a DataFrame.
        df = pd.DataFrame(input_data)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Makes a prediction using the loaded model.

    Args:
        input_data (pandas.DataFrame): The input data for prediction.
        model (object): The loaded model object.

    Returns:
        numpy.ndarray: The prediction result.
    """
    print("Making a prediction.")
    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, accept):
    """
    Serializes the prediction result into a JSON format.

    Args:
        prediction (numpy.ndarray): The prediction result from predict_fn.
        accept (str): The desired content type for the response.

    Returns:
        str: The serialized prediction result in JSON format.
    """
    if accept == "application/json":
        print("Serializing prediction to JSON.")
        return json.dumps(prediction.tolist()), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")