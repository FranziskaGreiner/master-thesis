import numpy as np
import pandas as pd
import wandb
import joblib
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import get_sarimax_config
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

sarimax_config = get_sarimax_config()


def transform_features(weather_time_moer_data):
    weather_time_moer_data['season_sin'] = np.sin(2 * np.pi * weather_time_moer_data['season'] / 4)
    weather_time_moer_data['season_cos'] = np.cos(2 * np.pi * weather_time_moer_data['season'] / 4)
    weather_time_moer_data['hour_of_day_sin'] = np.sin(2 * np.pi * weather_time_moer_data['hour_of_day'] / 24)
    weather_time_moer_data['hour_of_day_cos'] = np.cos(2 * np.pi * weather_time_moer_data['hour_of_day'] / 24)
    weather_time_moer_data['day_of_week_sin'] = np.sin(2 * np.pi * weather_time_moer_data['day_of_week'] / 7)
    weather_time_moer_data['day_of_week_cos'] = np.cos(2 * np.pi * weather_time_moer_data['day_of_week'] / 7)
    weather_time_moer_data['day_of_year_sin'] = np.sin(2 * np.pi * weather_time_moer_data['day_of_year'] / 365)
    weather_time_moer_data['day_of_year_cos'] = np.cos(2 * np.pi * weather_time_moer_data['day_of_year'] / 365)
    cols_to_drop = ['season', 'hour_of_day', 'day_of_week', 'day_of_year']
    weather_time_moer_data = weather_time_moer_data.drop(cols_to_drop, axis=1)
    return weather_time_moer_data


def normalize_features(weather_time_moer_data):
    scaler = StandardScaler()
    features_to_normalize = ['temperature', 'ghi', 'wind_speed', 'precipitation']
    scaler.fit(weather_time_moer_data[features_to_normalize])
    weather_time_moer_data.loc[:, features_to_normalize] = scaler.transform(
        weather_time_moer_data[features_to_normalize])
    return weather_time_moer_data


def interpolate_data(country_data, country):
    country_data.set_index('date', inplace=True)
    country_data.index = pd.to_datetime(country_data.index)
    country_data = country_data.asfreq('h')

    # Set country bc strings cannot be interpolated
    country_data['country'] = country
    country_data_complete = country_data.interpolate(method='linear', limit_direction='both')
    return country_data_complete


def create_sarimax_datasets(country_data_complete):
    country_data_complete = country_data_complete[(country_data_complete.index >= sarimax_config.get('start_date'))]
    train_data = country_data_complete[country_data_complete.index <= sarimax_config.get('training_cutoff_date')]
    validation_data = country_data_complete[country_data_complete.index > sarimax_config.get('training_cutoff_date')]

    exog_variables = [
        'temperature', 'ghi', 'wind_speed', 'precipitation', 'hour_of_day', 'day_of_week', 'day_of_year',
        'is_holiday_or_weekend', 'season'
    ]
    exog_train = train_data[exog_variables] if not train_data.empty else None
    exog_validation = validation_data[exog_variables] if not validation_data.empty else None
    return train_data, validation_data, exog_train, exog_validation


def create_auto_arima(train_data, exog_train):
    auto_arima_config = sarimax_config.get("auto_arima")
    sarimax_model = pm.auto_arima(train_data['moer'],
                                  exogenous=exog_train,
                                  start_p=auto_arima_config.get('start_p'), start_q=auto_arima_config.get('start_q'),
                                  max_p=auto_arima_config.get('max_p'), max_q=auto_arima_config.get('max_q'),
                                  m=auto_arima_config.get('m'),
                                  d=auto_arima_config.get('d'),
                                  seasonal=auto_arima_config.get('seasonal'),
                                  trace=auto_arima_config.get('trace'),
                                  suppress_warnings=auto_arima_config.get('suppress_warnings'),
                                  stepwise=auto_arima_config.get('stepwise'))
    print(sarimax_model.summary())


def create_sarimax_model(train_data, exog_train, config):
    sarimax_model = SARIMAX(train_data['moer'],
                            exog=exog_train,
                            order=(config.get('p'), config.get('d'), config.get('q')),
                            seasonal_order=(
                                config.get('P'), config.get('D'), config.get('Q'),
                                config.get('s')))
    return sarimax_model


def calculate_metrics(validation_data, val_predictions, country):
        # overall metrics
        val_mse = mean_squared_error(validation_data['moer'], val_predictions)
        val_mae = mean_absolute_error(validation_data['moer'], val_predictions)
        val_mape = mean_absolute_percentage_error(validation_data['moer'], val_predictions)
        wandb.log({f"{country}_validation_MSE": val_mse,
                   f"{country}_validation_MAE": val_mae,
                   f"{country}_validation_MAPE": val_mape})

        # weekly metrics
        # weekly_metrics = calculate_weekly_metrics(validation_data['moer'], val_predictions)
        # for week, metrics in weekly_metrics.items():
        #     wandb.log({f"{country}_{week}_validation_MSE": metrics['MSE'],
        #                f"{country}_{week}_validation_MAE": metrics['MAE'],
        #                f"{country}_{week}_validation_MAPE": metrics['MAPE']})


def calculate_weekly_metrics(actual, predicted):
    predicted = pd.Series(predicted, index=actual.index)

    actual_grouped = actual.groupby(actual.index.to_period('W'))
    predicted_grouped = predicted.groupby(predicted.index.to_period('W'))

    weekly_metrics = {}

    for (period, actual_values) in actual_grouped:
        if period in predicted_grouped.groups:  # check if the period exists in predicted groups
            predicted_values = predicted_grouped.get_group(period)

            if len(actual_values) == len(predicted_values):
                weekly_mse = mean_squared_error(actual_values, predicted_values)
                weekly_mae = mean_absolute_error(actual_values, predicted_values)
                weekly_mape = mean_absolute_percentage_error(actual_values, predicted_values)

                weekly_metrics[period] = {
                    'MSE': weekly_mse,
                    'MAE': weekly_mae,
                    'MAPE': weekly_mape
                }
        else:
            print(f"No predictions available for {period}")

    return weekly_metrics


def plot_evaluation(validation_data, val_predictions, country):
    plt.figure(figsize=(10, 5))
    plt.plot(validation_data.index, validation_data['moer'], label='Actual')
    plt.plot(validation_data.index, val_predictions, label='Prediction', linestyle='--')
    plt.xticks(rotation=45)
    plt.title(f'validation prediction vs. actual ({country})')
    plt.legend()
    plt.ylabel('MOER (gCO2/kWh)')
    plt.tight_layout()
    val_plot_file_path = f"{wandb.run.dir}/plot_val_{country}.png"
    plt.savefig(val_plot_file_path)
    plt.close()
    wandb.log({f"plot_val_{country}": wandb.Image(val_plot_file_path)})


def log_diagnostics(results, country):
    fig = results.plot_diagnostics(figsize=(10, 8))
    diagnostics_path = f"{wandb.run.dir}/diagnostics_{country}.png"
    fig.savefig(diagnostics_path)
    wandb.log({f"diagnostics_{country}": wandb.Image(diagnostics_path)})


def train_sarimax(weather_time_moer_data):
    run = wandb.init(project="tsf_moer_sarimax", config=sarimax_config)
    # weather_time_moer_data = transform_features(weather_time_moer_data)
    weather_time_moer_data = normalize_features(weather_time_moer_data)

    for country in ['DE', 'NO']:
        sarimax_country_config = sarimax_config.get(country.lower())
        country_data = weather_time_moer_data.loc[weather_time_moer_data['country'] == country]
        country_data_complete = interpolate_data(country_data, country)
        train_data, validation_data, exog_train, exog_validation = create_sarimax_datasets(country_data_complete)
        # create_auto_arima(train_data, exog_train)
        sarimax_model = create_sarimax_model(train_data, exog_train, sarimax_country_config)
        results = sarimax_model.fit(disp=False)

        print(results.summary())
        val_predictions = results.get_prediction(start=len(train_data),
                                                 end=len(train_data) + len(validation_data) - 1,
                                                 exog=exog_validation, dynamic=False).predicted_mean
        calculate_metrics(validation_data, val_predictions, country)
        plot_evaluation(validation_data, val_predictions, country)
        log_diagnostics(results, country)

        model_file_path = f"{wandb.run.dir}/sarimax_model_{country}.joblib"
        joblib.dump(results, model_file_path)
        artifact = wandb.Artifact(f'sarimax_model_{country}', type='model')
        artifact.add_file(model_file_path)
        run.log_artifact(artifact)

    run.finish()
