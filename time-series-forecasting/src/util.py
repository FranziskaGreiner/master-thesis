import os
import json
from pathlib import Path
from pandas import Timestamp
from .config import get_general_config

general_config = get_general_config()
output_path = general_config.get('output_path')


def check_if_preprocessed_data_exists():
    preprocessed_data_file_name = general_config.get('preprocessed_data_file_name')
    if os.getenv("COLAB_RELEASE_TAG"):
        data_path = '/content/drive/My Drive/data-collection/'
    else:
        data_path = general_config.get('data_path')
    preprocessed_data_file_path = os.path.join(data_path, preprocessed_data_file_name)
    return os.path.exists(preprocessed_data_file_path)


def save_config_and_results(run_dir, config, results):
    save_config(config, run_dir)
    # save_results(results, run_dir)


def save_config(config, run_dir):
    def default_handler(obj):
        if isinstance(obj, Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

    config_path = Path(run_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=default_handler)


def save_results(results, run_dir):
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
