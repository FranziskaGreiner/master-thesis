import pandas as pd
import wandb
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import get_sarimax_config
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sarimax_config = get_sarimax_config()


def create_sarimax_datasets(final_data):
    final_data.loc[:, 'date'] = pd.to_datetime(final_data['date'])
    final_data.set_index('date', inplace=True)
    final_data.index = pd.DatetimeIndex(final_data.index).to_period('H')

    train_data = final_data[:sarimax_config.get('training_cutoff_date')]
    validation_data = final_data[
                      sarimax_config.get('training_cutoff_date'):sarimax_config.get('validation_cutoff_date')]
    test_data = final_data[sarimax_config.get('validation_cutoff_date'):]

    # all feature variables ??
    exog_variables = ['temperature', 'radiation', 'wind_speed', 'day_of_week', 'is_holiday', 'season']
    exog_train = train_data[exog_variables] if not train_data.empty else None
    exog_validation = validation_data[exog_variables] if not validation_data.empty else None
    exog_test = test_data[exog_variables] if not test_data.empty else None
    return train_data, validation_data, test_data, exog_train, exog_validation, exog_test


def create_sarimax_model(train_data, exog_train):
    sarimax_model = SARIMAX(train_data['moer'],
                            exog=exog_train,
                            order=(sarimax_config.get('p'), sarimax_config.get('d'), sarimax_config.get('q')),
                            seasonal_order=(
                                sarimax_config.get('P'), sarimax_config.get('D'), sarimax_config.get('Q'),
                                sarimax_config.get('s')))
    return sarimax_model


def train_sarimax(final_data):
    run = wandb.init(project="tsf_sarimax", config=sarimax_config)
    models_results = {}

    for country in ['DE', 'SE']:
        country_data = final_data.loc[final_data['country'] == country]
        train_data, validation_data, test_data, exog_train, exog_validation, exog_test = create_sarimax_datasets(
            country_data)
        sarimax_model = create_sarimax_model(train_data, exog_train)
        results = sarimax_model.fit(disp=False)

        model_file_path = f"{wandb.run.dir}/sarimax_model_{country}.joblib"
        joblib.dump(results, model_file_path)
        artifact = wandb.Artifact(f'sarimax_model_{country}', type='model')
        artifact.add_file(model_file_path)
        run.log_artifact(artifact)

        if not validation_data.empty:
            val_predictions = results.get_prediction(start=len(train_data),
                                                     end=len(train_data) + len(validation_data) - 1,
                                                     exog=exog_validation, dynamic=False).predicted_mean
            val_mse = mean_squared_error(validation_data['moer'], val_predictions)
            val_mae = mean_absolute_error(validation_data['moer'], val_predictions)
            wandb.log({f"{country}_validation_MSE": val_mse, f"{country}_validation_MAE": val_mae})

        # if not test_data.empty:
        #     test_predictions = results.get_prediction(start=len(train_data) + len(validation_data),
        #     end= len(train_data) + len(validation_data) + len(test_data) - 1, exog=exog_test).predicted_mean
        #     test_mse = mean_squared_error(test_data['moer'], test_predictions)
        #     test_mae = mean_absolute_error(test_data['moer'], test_predictions)
        #     wandb.log({f"{country}_test_MSE": test_mse, f"{country}_test_MAE": test_mae})

        # Log diagnostics
        fig = results.plot_diagnostics(figsize=(10, 8))
        diagnostics_path = f"{wandb.run.dir}/diagnostics_{country}.png"
        fig.savefig(diagnostics_path)
        wandb.log({f"diagnostics_{country}": wandb.Image(diagnostics_path)})

    run.finish()
