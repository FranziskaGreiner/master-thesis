import pandas as pd
import os


def get_general_config():
    return {
        "data_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/'),
        "output_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out/'),
        "preprocessed_data_file_name": "weather_time_moer_2021_2023_DE_NO.csv",
        "training_cutoff_date": pd.to_datetime('2023-09-30 23:00'),
        "validation_cutoff_date": pd.to_datetime('2023-11-30 23:00'),
    }


def get_tft_config():
    general_config = get_general_config()
    tft_specific_config = {
        "gradient_clip_val": 0.1,
        "time_idx": "time_idx",
        "target": "moer",
        "group_ids": ["country"],
        "max_encoder_length": 168,  # 1 week
        "max_prediction_length": 168,  # 1 week
        "static_categoricals": ["country"],
        "time_varying_known_categoricals": ["season", "day_of_week", "is_holiday"],
        "time_varying_known_reals": ["ghi", "temperature", "wind_speed", "precipitation"],
        "time_varying_unknown_reals": ["moer"],
        "lags": {'moer': [24, 168]},
        "add_relative_time_idx": True,
        "add_target_scales": True,
        "add_encoder_length": True,
        "allow_missing_timesteps": True,
        "batch_size": 32,
        "num_workers": 2,
        # "max_epochs": 20,
        "accelerator:": "auto",
        "enable_model_summary": True,
        "learning_rate": 0.02,
        "hidden_size": 32,
        "attention_head_size": 1,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "log_interval": 10,
        "reduce_on_plateau_patience": 4,
    }
    return {**general_config, **tft_specific_config}


def get_sarimax_config():
    general_config = get_general_config()
    sarimax_config = {
        "de": {
            # SARIMAX parameters
            "p": 1, "d": 0, "q": 0,
            "P": 2, "D": 0, "Q": 0, "s": 24,

            # auto_arima parameters
            "start_p": 0, "start_q": 0, "max_p": 24, "max_q": 24, "m": 24, "seasonal": True,
            "test": "adf", "trace": True, "suppress_warnings": True, "stepwise": True,
        },
        "no": {
            # SARIMAX parameters
            "p": 1, "d": 0, "q": 0,
            "P": 2, "D": 0, "Q": 1, "s": 24,

            # auto_arima parameters
            "start_p": 0, "start_q": 0, "max_p": 24, "max_q": 24, "m": 24, "seasonal": True,
            "test": "adf", "trace": True, "suppress_warnings": True, "stepwise": True,
        }
    }
    return {**general_config, **sarimax_config}
