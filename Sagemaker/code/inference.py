from io import StringIO
import json
import logging
import os
import pandas as pd
from joblib import load
import numpy as np

logging.basicConfig(level=logging.INFO)

def model_fn(model_dir):
    logging.info("Files in model_dir:")
    for fname in os.listdir(model_dir):
        logging.info(fname)

    model_path = os.path.join(model_dir, 'model.joblib')
    model = load(model_path)
    return model

# Define input data processing function
def input_fn(request_body, request_content_type):
    """
    Process the incoming request data. Assume the data is a CSV without a header.
    """
    if request_content_type == 'text/csv':
        df = pd.read_csv(StringIO(request_body), header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Define the prediction function
def predict_fn(input_data, model):
    """
    Generate predictions from the input data using the loaded model.
    """
    # Assuming your model pipeline includes a StandardScaler and SFS, input_data can be used directly.
    # If additional preprocessing is required, perform it here.
    predictions = model.predict(input_data)
    return predictions

# # Define the output processing function
# def output_fn(prediction, accept='application/json'):
#     """
#     Format the predictions as JSON.
#     """
#     if accept == 'application/json':
#         return json.dumps({prediction.tolist()}), accept
#     else:
#         raise ValueError(f"Unsupported accept type: {accept}")
        
def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        # Convert NumPy arrays to list before serialization
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

