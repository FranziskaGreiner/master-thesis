import joblib
import pandas as pd
import torch
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime, timedelta
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from config import get_tft_config
from config import get_sarimax_config
from config import get_general_config

app = Flask(__name__)
tft_config = get_tft_config()
sarimax_config = get_sarimax_config()
general_config = get_general_config()
tft_model_path = f"{tft_config.get('output_path')}/tft_model.pth"
checkpoint_path = f"{tft_config.get('output_path')}/tft-epoch=09-val_loss=3.16.ckpt"
tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
exog_variables = ['temperature', 'ghi', 'wind_speed', 'precipitation', 'day_of_week', 'is_holiday', 'season']

de_start_date = datetime.strptime('2022-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
se_start_date = datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_type = data.get('model_type', 'tft')  # default: tft
    duration = data['duration']  # in minutes
    due_date = datetime.strptime(data['due_date'], '%Y-%m-%d %H:%M:%S')  # format: 'YYYY-MM-DD HH:MM:SS'
    country = data['country']

    # earliest_time_idx, latest_time_idx, data_for_prediction = get_time_indexes_and_data(due_date, duration, country)

    try:
        data_for_prediction = pd.read_csv(
            f"{general_config.get('data_path')}/weather_{country}_2024_2025_with_features.csv",
            index_col='date', parse_dates=True
        )
        validation_test_variables = pd.read_csv(
            f"{general_config.get('data_path')}/validation_test_{country}.csv",
            index_col='date', parse_dates=True
        )
        # data_for_prediction.index = pd.DatetimeIndex(data_for_prediction.index).to_period('H')
    except FileNotFoundError:
        return None, None, f"Data file not found for country {country}"

    exogenous_data = pd.concat([validation_test_variables, data_for_prediction[exog_variables]])
    exogenous_data.index = pd.to_datetime(exogenous_data.index)

    # Erstellen eines vollständigen Datumsbereichs
    start_date = exogenous_data.index.min()
    end_date = exogenous_data.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Überprüfen, ob alle Zeitpunkte vorhanden sind
    missing_times = full_range.difference(exogenous_data.index)

    if missing_times.empty:
        print("Es fehlen keine stündlichen Daten.")
    else:
        print("Fehlende Zeitpunkte:", missing_times)

    if model_type == 'tft':
        optimal_start_time = find_optimal_start_time(
            model_type, tft_model, duration, None, None, exogenous_data
        )
    elif model_type == 'sarimax':
        sarimax_model_path = f"{sarimax_config.get('output_path')}/sarimax_model_{country}.joblib"
        sarimax_model = joblib.load(sarimax_model_path)
        optimal_start_time = find_optimal_start_time(
            model_type, sarimax_model, duration, datetime.now(), due_date, exogenous_data
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


def find_optimal_start_time(model_type, model, duration, earliest_time, latest_time, exog_data):
    best_time_slot = None
    lowest_moer_sum = float('inf')

    # Iterate over possible start times from now until the latest start time
    daterange = pd.date_range(start=earliest_time, end=latest_time, freq='H')
    for date in daterange:
        end_time = date + timedelta(minutes=duration)
        if model_type == 'sarimax':
            predicted_moer = predict_with_sarimax(model, date, end_time, exog_data)
            moer_sum = np.sum(predicted_moer[:duration // 60])  # prediction is hourly and duration is in minutes

            if moer_sum < lowest_moer_sum:
                best_time_slot = date
                lowest_moer_sum = moer_sum
        else:
            predicted_moer = predict_with_tft(model, date, duration, exog_data)
            moer_sum = np.sum(predicted_moer[:duration // 60])  # prediction is hourly and duration is in minutes

            if moer_sum < lowest_moer_sum:
                best_time_slot = date
                lowest_moer_sum = moer_sum

    optimal_start_time = exog_data[exog_data.index == best_time_slot].iloc[0]
    return optimal_start_time.strftime('%Y-%m-%d %H:%M:%S')


def predict_with_tft(model, start_time_idx, duration, data):
    scaler = joblib.load(f"{general_config.get('output_path')}feature_scaler.joblib")
    features_to_normalize = ['temperature', 'ghi', 'precipitation', 'wind_speed']
    target_normalizer = GroupNormalizer(groups=["country"], transformation="softplus")

    data[features_to_normalize] = scaler.transform(data[features_to_normalize])
    data = data[data["country"] == "DE"]
    data['is_holiday'] = data['is_holiday'].astype(str)
    data['season'] = data['season'].astype('category')
    data['day_of_week'] = data['day_of_week'].astype('category')

    # Ensure that moer is not part of the data
    if 'moer' in data.columns:
        data = data.drop('moer', axis=1)

    training_dataset_params = torch.load(f"{tft_config.get('output_path')}/training_dataset_params.pth")
    training_dataset_params['time_varying_unknown_reals'] = []
    training_dataset_params['target'] = ''
    training_dataset_params['target_normalizer'] = target_normalizer

    training_dataset_params['time_varying_known_reals'] = [
        item for item in training_dataset_params['time_varying_known_reals']
        if not item.startswith('moer')
    ]

    prediction_dataset = TimeSeriesDataSet.from_parameters(
        parameters=training_dataset_params,
        data=data,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=start_time_idx,
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


def predict_with_sarimax(model, start_time, end_time, data):
    exog_data = data.loc[:end_time]
    start = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end = end_time.strftime('%Y-%m-%d %H:%M:%S')
    predicted_moer = model.predict(start=start, end=end, exog=exog_data)
    return predicted_moer


if __name__ == "__main__":
    app.run()
