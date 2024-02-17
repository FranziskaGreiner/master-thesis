import pandas as pd
import os


def get_general_config():
    return {
        "data_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/'),
        "output_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out/'),
        "preprocessed_data_file_name": "combined_weather_moer_2021_2023_with_features.csv",
        "training_cutoff_date": pd.Timestamp('2023-09-30'),
        "validation_cutoff_date": pd.Timestamp('2023-11-30'),
    }


def get_tft_config():
    general_config = get_general_config()
    tft_specific_config = {
        "max_encoder_length": 168,
        "max_prediction_length": 168,
        "lags": [24, 168, 8736],
        "batch_size": 32,
        "num_workers": 2,
        "max_epochs": 10,
        "accelerator:": "auto",
        "enable_model_summary": True,
        "learning_rate": 0.01,
        "hidden_size": 32,
        "attention_head_size": 4,
        "dropout": 0.4,
        "hidden_continuous_size": 8,
        "output_size": 7,
        "log_interval": 10,
        "reduce_on_plateau_patience": 2,
    }
    return {**general_config, **tft_specific_config}


def get_sarimax_config():
    general_config = get_general_config()
    sarimax_specific_config = {
        # Direct SARIMAX parameters
        "p": 1, "d": 1, "q": 1,  # Non-seasonal order
        "P": 1, "D": 1, "Q": 1, "s": 12,  # Seasonal order

        # Parameters for auto_arima (if used for model selection)
        "start_p": 1, "start_q": 1, "max_p": 3, "max_q": 3, "m": 12,
        "start_P": 0, "seasonal": True,
        "trace": False, "error_action": 'ignore', "suppress_warnings": True, "stepwise": True,
    }
    return {**general_config, **sarimax_specific_config}
