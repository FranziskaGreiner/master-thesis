import os
import json
import pandas as pd
from pathlib import Path
from pandas import Timestamp
from .config import get_general_config

general_config = get_general_config()
output_path = general_config.get('output_path')


def get_preprocessed_data(filtered):
    if filtered:
        preprocessed_data_file_name = general_config.get('preprocessed_filtered_data_file_name')
    else:
        preprocessed_data_file_name = general_config.get('preprocessed_data_file_name')
    if os.getenv("COLAB_RELEASE_TAG"):
        data_path = '/content/drive/My Drive/data_collection/'
        print('running in colab')
    else:
        data_path = general_config.get('data_path')
    return pd.read_csv(data_path + preprocessed_data_file_name)
