import pandas as pd
import argparse
import wandb

from src.config import get_general_config, get_tft_config, get_sarimax_config
from src.util import check_if_preprocessed_data_exists
from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.models.sarimax import train_sarimax
from src.models.tft import train_tft


def main(model_type):
    general_config = get_general_config()
    preprocessed_feature_engineered_data_exists = check_if_preprocessed_data_exists()
    wandb.login(key="361ced3122f96ccbe37b41a4ec49c643503bc408")

    if not preprocessed_feature_engineered_data_exists:
        print('preprocessing data')
        # 1. data preprocessing
        preprocessed_data = preprocess_data()
        # 2. feature engineering
        preprocessed_feature_engineered_data = add_features(preprocessed_data)
    else:
        print('using existing data file')
        preprocessed_feature_engineered_data = pd.read_csv(general_config.get('data_path') + general_config.get('preprocessed_data_file_name'))

    # 3. training and 4. evaluation
    if model_type == 'sarimax':
        sarimax_model = train_sarimax(preprocessed_feature_engineered_data)
        # evaluate_sarimax(sarimax_model, preprocessed_feature_engineered_data)
    elif model_type == 'tft':
        tft_model = train_tft(preprocessed_feature_engineered_data)
        # evaluate_tft(tft_model, preprocessed_feature_engineered_data)
    else:
        raise ValueError("Unsupported model type specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time series forecasting models with TFT or SARIMAX")
    parser.add_argument('--model_type', type=str, choices=['sarimax', 'tft'], help="The type of model to train and evaluate.")
    args = parser.parse_args()

    main(args.model_type)
