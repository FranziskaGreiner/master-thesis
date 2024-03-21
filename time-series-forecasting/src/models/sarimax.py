import pandas as pd
import wandb
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import get_sarimax_config
from sklearn.metrics import mean_squared_error, mean_absolute_error

sarimax_config = get_sarimax_config()


def create_sarimax_datasets(final_data):
    final_data.loc[:, 'date'] = pd.to_datetime(final_data['date'])
    final_data.set_index('date', inplace=True)
    final_data.index = pd.DatetimeIndex(final_data.index)

    train_data = final_data[:sarimax_config.get('training_cutoff_date')]
    validation_data = final_data[
                      sarimax_config.get('training_cutoff_date'):sarimax_config.get('validation_cutoff_date')]
    test_data = final_data[sarimax_config.get('validation_cutoff_date'):]

    exog_variables = ['temperature', 'ghi', 'wind_speed', 'precipitation', 'day_of_week', 'is_holiday', 'season']
    exog_train = train_data[exog_variables] if not train_data.empty else None
    exog_validation = validation_data[exog_variables] if not validation_data.empty else None
    exog_test = test_data[exog_variables] if not test_data.empty else None
    return train_data, validation_data, test_data, exog_train, exog_validation, exog_test


def create_sarimax_model(train_data, exog_train, config):
    sarimax_model = SARIMAX(train_data['moer'],
                            exog=exog_train,
                            order=(config.get('p'), config.get('d'), config.get('q')),
                            seasonal_order=(
                                config.get('P'), config.get('D'), config.get('Q'),
                                config.get('s')))
    return sarimax_model


def train_sarimax(final_data):
    run = wandb.init(project="tsf_moer_sarimax", config=sarimax_config)

    for country in ['DE', 'NO']:
        sarimax_country_config = sarimax_config.get(country.lower())
        country_data = final_data.loc[final_data['country'] == country]
        train_data, validation_data, test_data, exog_train, exog_validation, exog_test = create_sarimax_datasets(
            country_data)
        sarimax_model = create_sarimax_model(train_data, exog_train, sarimax_country_config)
        results = sarimax_model.fit(disp=False)

        # model_file_path = f"{wandb.run.dir}/sarimax_model_{country}.joblib"
        # joblib.dump(results, model_file_path)
        # artifact = wandb.Artifact(f'sarimax_model_{country}', type='model')
        # artifact.add_file(model_file_path)
        # run.log_artifact(artifact)

        if not validation_data.empty:
            val_predictions = results.get_prediction(start=len(train_data),
                                                     end=len(train_data) + len(validation_data) - 1,
                                                     exog=exog_validation, dynamic=False).predicted_mean
            val_mse = mean_squared_error(validation_data['moer'], val_predictions)
            val_mae = mean_absolute_error(validation_data['moer'], val_predictions)
            wandb.log({f"{country}_validation_MSE": val_mse, f"{country}_validation_MAE": val_mae})

            plt.figure(figsize=(10, 5))
            plt.plot(validation_data.index, validation_data['moer'], label='Actual')
            plt.plot(validation_data.index, val_predictions, label='Prediction', linestyle='--')
            plt.xticks(rotation=30)
            plt.title(f'Validation Prediction vs Actual for {country}')
            plt.legend()
            val_plot_file_path = f"{wandb.run.dir}/plot_val_{country}.png"
            plt.savefig(val_plot_file_path)
            plt.close()
            wandb.log({f"plot_val_{country}": wandb.Image(val_plot_file_path)})

        # if not test_data.empty:

        # Log diagnostics
        fig = results.plot_diagnostics(figsize=(10, 8))
        diagnostics_path = f"{wandb.run.dir}/diagnostics_{country}.png"
        fig.savefig(diagnostics_path)
        wandb.log({f"diagnostics_{country}": wandb.Image(diagnostics_path)})

    run.finish()
