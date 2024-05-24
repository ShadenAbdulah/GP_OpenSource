import os
import pandas as pd
import json 
import boto3 
import io
from features import FeatureEng
import time


time.sleep(3)
s3 = boto3.client('s3')

def check_folder_exists(prefix):
    bucket = 'nabeeh'
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response

def load_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f"Error loading {key} from S3: {e}")
        return None

def process_sample(path):
    bucket_name = 'nabeeh'
    start_time = time.time()
    file_keys = [f'{path}{i}-ZL_trace.csv' for i in [1, 3, 5]] + \
                [f'{path}{i}-ZL_predict.csv' for i in [2, 4, 6]] + \
                [f'{path}{i}-PL_trace.csv' for i in [7, 9, 11]] + \
                [f'{path}{i}-PL_predict.csv' for i in [8, 10, 12]]

    data_frames = [load_from_s3(bucket_name, key) for key in file_keys]
    print('Loading and processing files took:', time.time() - start_time)

    ZL_trace_frames = []
    ZL_predict_frames = []
    PL_trace_frames = []
    PL_predict_frames = []

    for key, data in zip(file_keys, data_frames):
        if data is not None:
            file_number = int(key.split('/')[-1].split('-')[0])
            if file_number in [1, 3, 5]:
                ZL_trace_frames.append(FeatureEng(data))
            elif file_number in [2, 4, 6]:
                ZL_predict_frames.append(FeatureEng(data))
            elif file_number in [7, 9, 11]:
                PL_trace_frames.append(FeatureEng(data))
            elif file_number in [8, 10, 12]:
                PL_predict_frames.append(FeatureEng(data))

    print('Feature engineering took:', time.time() - start_time)

    # Concatenate all DataFrames if they are not empty
    def safe_concat(frames):
        if frames:
            return pd.concat(frames).reset_index(drop=True)
        else:
            return pd.DataFrame()

    ZL_trace = safe_concat(ZL_trace_frames)
    ZL_predict = safe_concat(ZL_predict_frames)
    PL_trace = safe_concat(PL_trace_frames)
    PL_predict = safe_concat(PL_predict_frames)

    return ZL_trace, ZL_predict, PL_trace, PL_predict

def result_fun(df, taskName, predictions): 
    model_files = {
        'ZL_trace': 'task1/model.tar.gz',
        'ZL_predict': 'task2/model.tar.gz',
        'PL_trace': 'task3/model.tar.gz',
        'PL_predict': 'task4/model.tar.gz'
    }

    feature_indices = {
        'ZL_trace': [6, 7, 20, 26, 28, 29],
        'ZL_predict': [2, 6, 8, 24],
        'PL_trace': [0, 1, 3, 6, 8, 14, 15, 21, 22, 23],
        'PL_predict':  [2, 3, 4, 6, 8, 10, 14, 23, 24, 26, 28]
    }
    
    model_name = model_files[taskName]
    x = df.iloc[:, feature_indices[taskName]]
    payload = x.to_csv(index=False, header=False).encode('utf-8')

    client = boto3.client('sagemaker-runtime')
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

    response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='text/csv',
                                      Body=payload,
                                      TargetModel=model_name)
    result = json.loads(response['Body'].read().decode())
    print('Done model', result)
    predictions.append(result)

def calculate_weighted_average(predictions):
    accuracies = [91.8, 88.6, 90.2, 95.9]
    weighted_sum = 0
    total_weight = 0

    for i, preds in enumerate(predictions):
        weight = accuracies[i] / 100  # Convert accuracy to weight
        weighted_sum += sum(preds) * weight
        total_weight += weight * len(preds)

    weighted_average = weighted_sum / total_weight
    return weighted_average

def lambda_handler(event, context):
    try:
        start_time = time.time()
        query_params = event.get('queryStringParameters', {})
        user_id = query_params.get('userID')

        if user_id is not None and int(user_id) > 0:
            path = f"system_users/{user_id}/"
            folder_exists = check_folder_exists(path)
            print(folder_exists)
            if not folder_exists:
                return {
                    'statusCode': 404,
                    'body': f'Folder for user ID {user_id} does not exist in S3.'
                }
                
            file_key = f"{path}/output.json"
            file_exists = check_folder_exists(file_key)
            if file_exists:
                # Read JSON data from S3
                response = s3.get_object(Bucket='nabeeh', Key=file_key)
                json_data = response['Body'].read().decode('utf-8')
                data = json.loads(json_data)
                print('it exist:',data)
                print('it exist takes: ',time.time() - start_time)

                return {
                    'statusCode': 200,
                    'body': json.dumps(data)
                }
                
            ZL_trace, ZL_predict, PL_trace, PL_predict = process_sample(path)
            predictions = []
            
            result_fun(ZL_trace, 'ZL_trace', predictions)
            result_fun(ZL_predict, 'ZL_predict', predictions)
            result_fun(PL_trace, 'PL_trace', predictions)
            result_fun(PL_predict, 'PL_predict', predictions)

            result = calculate_weighted_average(predictions)
            r = round(result, 2)

            elapsed_time = time.time() - start_time
            print('Total execution time:', elapsed_time)

            json_data = json.dumps(int(r * 100))
            s3.put_object(Bucket='nabeeh', Key=file_key, Body=json_data)

            return {
                'statusCode': 200,
                'body': json_data
            }
        return {
            'statusCode': 500,
            'body': json.dumps(f'Invalid user ID: {user_id}')
        }    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }