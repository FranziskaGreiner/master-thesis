import joblib
import pandas as pd
import torch
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime, timedelta
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from config import get_tft_config
from config import get_sarimax_config
from config import get_general_config

app = Flask(__name__)
tft_config = get_tft_config()
sarimax_config = get_sarimax_config()
general_config = get_general_config()
tft_model_path = f"{tft_config.get('output_path')}/tft_model.pth"
sarimax_model_path = f"{sarimax_config.get('output_path')}/sarimax_model.pth"
sarimax_model = SARIMAXResults.load(sarimax_model_path)

checkpoint_path = f"{tft_config.get('output_path')}/tft-epoch=09-val_loss=3.16.ckpt"
tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

de_start_date = datetime.strptime('2022-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
se_start_date = datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_type = data.get('model_type', 'tft')  # default: tft
    duration = data['duration']  # in minutes
    due_date = datetime.strptime(data['due_date'], '%Y-%m-%d_%H:%M:%S')  # format: 'YYYY-MM-DD HH:MM:SS'
    country = data['country']

    earliest_time_idx, latest_time_idx, data_for_prediction = get_time_indexes_and_data(due_date, duration, country)

    if model_type == 'tft':
        optimal_start_time = find_optimal_start_time(
            model_type, tft_model, duration, earliest_time_idx, latest_time_idx, data_for_prediction
        )
    elif model_type == 'sarimax':
        optimal_start_time = find_optimal_start_time(
            model_type, sarimax_model, duration, earliest_time_idx, latest_time_idx, data_for_prediction
        )
    else:
        return jsonify("No valid model type specified, using default: tft")

    return jsonify({'optimal_start_time': optimal_start_time})


def get_time_indexes_and_data(due_date, duration, country):
    earliest_start_time = datetime.now()
    latest_start_time = due_date - timedelta(minutes=duration)

    if country == 'DE':
        earliest_time_idx = int((earliest_start_time - de_start_date).total_seconds() // 3600)
        latest_time_idx = int((latest_start_time - de_start_date).total_seconds() // 3600)
    elif country == 'SE':
        earliest_time_idx = int((earliest_start_time - se_start_date).total_seconds() // 3600)
        latest_time_idx = int((latest_start_time - se_start_date).total_seconds() // 3600)

    try:
        data_for_prediction = pd.read_csv(f"{general_config.get('data_path')}/weather_{country}_2024_2025_with_features.csv")
    except FileNotFoundError:
        return None, None, f"Data file not found for country {country}"

    return earliest_time_idx, latest_time_idx, data_for_prediction


def find_optimal_start_time(model_type, model, duration, earliest_time_idx, latest_time_idx, data):
    best_time_slot = None
    lowest_moer_sum = float('inf')

    # Iterate over possible start times from now until the latest start time
    for time_idx in range(earliest_time_idx, latest_time_idx):
        if model_type == 'sarimax':
            predicted_moer = predict_with_sarimax(model, time_idx, time_idx + duration // 60, data)
            moer_sum = np.sum(predicted_moer[:duration // 60])  # prediction is hourly and duration is in minutes

            if moer_sum < lowest_moer_sum:
                best_time_slot = time_idx
                lowest_moer_sum = moer_sum
        else:
            predicted_moer = predict_with_tft(model, time_idx, duration, data)
            moer_sum = np.sum(predicted_moer[:duration // 60])  # prediction is hourly and duration is in minutes

            if moer_sum < lowest_moer_sum:
                best_time_slot = time_idx
                lowest_moer_sum = moer_sum

    optimal_start_time = data[data['time_idx'] == best_time_slot]['date'].iloc[0]
    return optimal_start_time.strftime('%Y-%m-%d %H:%M:%S')


def predict_with_tft(model, start_time_idx, duration, data):
    scaler = joblib.load(f"{general_config.get('output_path')}feature_scaler.joblib")
    features_to_normalize = ['temperature', 'radiation', 'wind_speed']
    data[features_to_normalize] = scaler.transform(data[features_to_normalize])
    data['is_holiday'] = data['is_holiday'].astype(str)
    data['season'] = data['season'].astype('category')
    data['day_of_week'] = data['day_of_week'].astype('category')
    data['time_idx'] = (data['time_idx']).astype(float)

    training_dataset_params = torch.load(f"{tft_config.get('output_path')}/training_dataset_params.pth")

    prediction_dataset = TimeSeriesDataSet.from_parameters(
        parameters=training_dataset_params,
        data=data,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=data[start_time_idx],
        max_prediction_length=duration
    )
    predict_dataloader = prediction_dataset.to_dataloader(batch_size=320, train=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in predict_dataloader:
            batch_prediction = model.predict(batch, mode="prediction")
            predictions.append(batch_prediction.numpy())
    return predictions


def predict_with_sarimax(model, start_time_idx, end_time_idx, data):
    exog_variables = ['temperature', 'radiation', 'wind_speed']
    exog_data = data[exog_variables]
    predicted_moer = model.predict(start=start_time_idx, end=end_time_idx, exog=exog_data)
    return predicted_moer


if __name__ == "__main__":
    app.run(debug=True)
