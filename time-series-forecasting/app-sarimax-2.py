import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime, timedelta
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from config import get_tft_config
from config import get_sarimax_config
from config import get_general_config

app = Flask(__name__)
tft_config = get_tft_config()
sarimax_config = get_sarimax_config()
general_config = get_general_config()
tft_model_path = f"{tft_config.get('output_path')}/tft_model.pth"
checkpoint_path = f"{tft_config.get('output_path')}/tft-epoch=04-val_loss=8.93.ckpt"
tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
exog_variables = [
    'temperature', 'ghi', 'wind_speed', 'precipitation', 'hour_of_day', 'day_of_week',
    'day_of_year', 'is_holiday_or_weekend', 'season'
]

de_start_date = pd.to_datetime('2022-10-01')
no_start_date = pd.to_datetime('2021-01-01')


@app.route('/predict/past', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_type = data.get('model_type', 'tft')  # default: tft
    prediction_length = data.get('prediction_length', 168)  # 168: 1w, 720: 1m, default: 1w
    duration = data['duration']  # in minutes
    due_date = datetime.strptime(data['due_date'], '%Y-%m-%d %H:%M:%S')  # format: 'YYYY-MM-DD HH:MM:SS'
    country = data['country']

    try:
        data_for_prediction = pd.read_csv(
            f"{general_config.get('data_path')}/weather_time_{country}_2024-2025.csv",
            index_col='date', parse_dates=True
        )
    except FileNotFoundError:
        return None, None, f"Data file not found for country {country}"

    exogenous_data = data_for_prediction[exog_variables]
    exogenous_data.index = pd.to_datetime(exogenous_data.index)

    first_day_of_prediction = pd.to_datetime('2024-02-01') if prediction_length == 720 else pd.to_datetime('2024-02-23')
    earliest_time = first_day_of_prediction
    latest_time = due_date - timedelta(minutes=duration)

    if model_type == 'sarimax':
        sarimax_model_path = f"{sarimax_config.get('output_path')}/sarimax_model_{country}_{prediction_length}.joblib"
        sarimax_model = joblib.load(sarimax_model_path)
        optimal_start_time, lowest_moer_sum, highest_moer_sum = find_optimal_start_time(
            model_type, sarimax_model, duration, earliest_time, latest_time, exogenous_data, country
        )
    else:
        return jsonify("No valid model type specified, use either tft or sarimax")

    if optimal_start_time is None or lowest_moer_sum == 0:
        return "There was an error processing the request"
    else:
        optimal_start_time = optimal_start_time.replace(second=0, microsecond=0)
        return f'Best start time is {optimal_start_time} with an expected MOER sum of {lowest_moer_sum.round(2)} g. ' \
               f'You can save up to {(highest_moer_sum - lowest_moer_sum).round(2)} g.'


def hour_rounder(date):
    return (date.replace(second=0, microsecond=0, minute=0, hour=date.hour)
            + timedelta(hours=date.minute // 30))


def find_optimal_start_time(model_type, model, duration, earliest_time, latest_time, exog_data, country):
    best_time_slot = None
    lowest_moer_sum = float('inf')
    highest_moer_sum = 0

    # Iterate over possible start times from now until the latest start time
    daterange = pd.date_range(start=earliest_time, end=latest_time, freq='h')
    for date in daterange:
        end_time = date + timedelta(minutes=duration)
        if model_type == 'sarimax':
            predicted_moer = predict_with_sarimax(model, earliest_time, date, end_time, exog_data)
            moer_sum = np.sum(predicted_moer[:duration // 60])  # prediction is hourly and duration is in minutes

            if (moer_sum < lowest_moer_sum) & (moer_sum > 0):
                best_time_slot = date
                lowest_moer_sum = moer_sum
            elif moer_sum > highest_moer_sum:
                highest_moer_sum = moer_sum

    # optimal_start_time = exog_data[exog_data.index == best_time_slot].iloc[0]
    return best_time_slot, lowest_moer_sum, highest_moer_sum


def predict_with_sarimax(model, earliest_time, start_time, end_time, data):
    exog_data = data[earliest_time:end_time]  # + timedelta(hours=1)
    start = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end = end_time.strftime('%Y-%m-%d %H:%M:%S')
    predicted_moer = model.predict(start=start, end=end, exog=exog_data)
    return predicted_moer


if __name__ == "__main__":
    app.run()
