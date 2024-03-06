import argparse
import wandb

from src.util import get_preprocessed_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.models.sarimax import train_sarimax
from src.models.tft import train_tft


def main(model_type):
    preprocessed_feature_engineered_data = get_preprocessed_data()
    wandb.login(key="361ced3122f96ccbe37b41a4ec49c643503bc408")

    if preprocessed_feature_engineered_data is None:
        print('preprocessing data')
        # 1. data preprocessing
        preprocessed_data = preprocess_data()
        # 2. feature engineering
        preprocessed_feature_engineered_data = add_features(preprocessed_data)
    else:
        print('using existing data file')

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
