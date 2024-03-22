import pandas as pd
import os


def get_general_config():
    return {
        "data_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/'),
        "output_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out/'),
        "preprocessed_data_file_name": "weather_time_moer_2021_2023_DE_NO.csv",
        "training_cutoff_date": pd.to_datetime('2023-11-30 23:00'),
    }


def get_tft_config():
    general_config = get_general_config()
    tft_specific_config = {
        "gradient_clip_val": 0.8,
        "time_idx": "time_idx",
        "target": "moer",
        "group_ids": ["country"],
        "max_encoder_length": 672,  # 4 weeks
        "max_prediction_length": 336,  # 2 week
        "static_categoricals": ["country"],
        "time_varying_known_categoricals": ["season", "day_of_week", "is_holiday"],
        "time_varying_known_reals": ["ghi", "temperature", "wind_speed", "precipitation"],
        "time_varying_unknown_reals": ["moer"],
        "lags": {'moer': [24, 168]},
        "add_relative_time_idx": True,
        "add_target_scales": True,
        "add_encoder_length": True,
        "allow_missing_timesteps": True,
        "batch_size": 16,
        "num_workers": 2,
        "max_epochs": 20,
        "accelerator:": "auto",
        "enable_model_summary": True,
        "learning_rate": 0.01,
        "hidden_size": 18,
        "attention_head_size": 3,
        "dropout": 0.2,
        "hidden_continuous_size": 14,
        "log_interval": 10,
        "reduce_on_plateau_patience": 5,
    }
    return {**general_config, **tft_specific_config}


def get_sarimax_config():
    general_config = get_general_config()
    sarimax_config = {
        "de": {
            "p": 3, "d": 0, "q": 0,
            "P": 2, "D": 0, "Q": 0, "s": 24,
        },
        "no": {
            "p": 3, "d": 0, "q": 0,
            "P": 2, "D": 0, "Q": 0, "s": 24,
        },
        "auto_arima": {
            "start_p": 0, "start_q": 0, "max_p": 3, "max_q": 3, "m": 24, "d": 0,
            "seasonal": True, "trace": True, "suppress_warnings": True, "stepwise": True,
        }
    }
    return {**general_config, **sarimax_config}
