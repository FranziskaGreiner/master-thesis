import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import get_sarimax_config
from src.util import create_run_directory, save_config_and_results

sarimax_config = get_sarimax_config()
run_dir = create_run_directory('sarimax')


def create_sarimax_datasets(final_data):
    final_data['date'] = pd.to_datetime(final_data['date'])

    train_data = final_data[final_data['date'] <= sarimax_config.get('training_cutoff_date')]
    validation_data = final_data[(final_data['date'] > sarimax_config.get('training_cutoff_date')) & (
                final_data['date'] <= sarimax_config.get('validation_cutoff_date'))]
    test_data = final_data[final_data['date'] > sarimax_config.get('validation_cutoff_date')]

    exog_variables = ['temperature', 'radiation', 'wind_speed']
    # first two radiation values are NaN, set them to 0
    train_data.loc[train_data.index[:2], 'radiation'] = train_data.loc[train_data.index[:2], 'radiation'].fillna(0)
    exog_train = train_data[exog_variables]

    exog_validation = validation_data[exog_variables] if not validation_data.empty else None
    exog_test = test_data[exog_variables] if not test_data.empty else None
    return train_data, validation_data, test_data, exog_train, exog_validation, exog_test


def create_sarimax_model(train_data, exog_train):
    sarimax_model = SARIMAX(train_data['moer'],
                            exog=exog_train,
                            order=(sarimax_config.get('p'), sarimax_config.get('d'), sarimax_config.get('q')),
                            seasonal_order=(
                            sarimax_config.get('P'), sarimax_config.get('D'), sarimax_config.get('Q'), sarimax_config.get('s')))
    save_config_and_results(run_dir, sarimax_config, None)
    return sarimax_model


def train_sarimax (final_data):
    train_data, validation_data, test_data, exog_train, exog_validation, exog_test = create_sarimax_datasets(final_data)
    sarimax_model = create_sarimax_model(train_data, exog_train)
    results = sarimax_model.fit()
    # disp=False makes the fitting silent, not printing any output
