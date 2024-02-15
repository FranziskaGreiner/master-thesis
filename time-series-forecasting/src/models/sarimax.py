import pandas as pd
import wandb
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import get_sarimax_config
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sarimax_config = get_sarimax_config()


def create_sarimax_datasets(final_data):
    final_data = final_data.copy()
    final_data['date'] = pd.to_datetime(final_data['date'])

    train_data = final_data[final_data['date'] <= sarimax_config.get('training_cutoff_date')].copy()
    train_data.loc[:, 'time_idx'] = train_data['time_idx'].astype(int)
    validation_data = final_data[
        (final_data['date'] > sarimax_config.get('training_cutoff_date')) &
        (final_data['date'] <= sarimax_config.get('validation_cutoff_date'))
        ].copy()
    validation_data.loc[:, 'time_idx'] = validation_data['time_idx'].astype(int)
    test_data = final_data[final_data['date'] > sarimax_config.get('validation_cutoff_date')].copy()
    test_data.loc[:, 'time_idx'] = test_data['time_idx'].astype(int)

    exog_variables = ['temperature', 'radiation', 'wind_speed']
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
    # save_config_and_results(run_dir, sarimax_config, None)
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
        models_results[country] = results

        model_file_path = f"{wandb.run.dir}/sarimax_model_{country}.joblib"
        joblib.dump(results, model_file_path)
        artifact = wandb.Artifact(f'sarimax_model_{country}', type='model')
        artifact.add_file(model_file_path)
        run.log_artifact(artifact)

        fig = results.plot_diagnostics(figsize=(10, 8))
        diagnostics_path = f"{wandb.run.dir}/diagnostics_{country}.png"
        fig.savefig(diagnostics_path)
        wandb.log({f"diagnostics_{country}": wandb.Image(diagnostics_path)})
        wandb.log({f"model_{country}_summary": results.summary().as_text()})

        if not validation_data.empty:
            val_metrics, val_predictions = predict_and_evaluate(validation_data, exog_validation, results)
            wandb.log({"validation_metrics": val_metrics})
            wandb.log({"val_predictions": wandb.Table(dataframe=val_predictions.to_frame(name='predictions'))})

        if not test_data.empty:
            test_metrics, test_predictions = predict_and_evaluate(test_data, exog_test, results)
            wandb.log({"test_metrics": test_metrics})
            wandb.log({"test_predictions": wandb.Table(dataframe=test_predictions.to_frame(name='predictions'))})

    run.finish()
    return models_results


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "R^2": r2}


def predict_and_evaluate(data, exog, results):
    predictions = results.get_prediction(start=data.index[0], end=data.index[-1], exog=exog, dynamic=False)
    pred_mean = predictions.predicted_mean
    metrics = calculate_metrics(data['moer'], pred_mean)
    return metrics, pred_mean
