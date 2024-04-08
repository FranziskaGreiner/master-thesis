import argparse
import wandb

from src.util import get_preprocessed_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.models.sarimax import train_sarimax
from src.models.tft import train_tft


def main(model_type, use_filtered_data):
    wandb.login(key="361ced3122f96ccbe37b41a4ec49c643503bc408")

    preprocessed_data = get_preprocessed_data(use_filtered_data)

    if preprocessed_data is None:
        print('preprocessing data')
        # 1. data preprocessing
        preprocessed_data = preprocess_data()
        # 2. feature engineering
        preprocessed_data = add_features(preprocessed_data)
    else:
        print('using existing data file')

    # 3. training and 4. evaluation
    if model_type == 'sarimax':
        sarimax_model = train_sarimax(preprocessed_data)
        # evaluate_sarimax(sarimax_model, preprocessed_data)
    elif model_type == 'tft':
        tft_model = train_tft(preprocessed_data)
        # evaluate_tft(tft_model, preprocessed_data)
    else:
        raise ValueError("Unsupported model type specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time series forecasting models with TFT or SARIMAX")
    parser.add_argument('--model_type', type=str, choices=['sarimax', 'tft'], help="The type of model to train and evaluate.")
    parser.add_argument('--use_filtered_data', type=bool, help="Whether to use the dataset where outliers are removed or the unfiltered dataset.")
    args = parser.parse_args()

    main(args.model_type, args.use_filtered_data)
