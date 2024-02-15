import numpy as np
import pandas as pd
import glob
import os
from src.config import get_general_config

data_path = get_general_config()["data_path"]
temperature_file = data_path + "H_ERA5_ECMW_T639_TA-_0002m_Euro_NUT0_S197901010000_E202312312300_INS_TIM_01h_NA-_noc_org_NA_NA---_NA---_NA---.csv"
radiation_file = data_path + "H_ERA5_ECMW_T639_GHI_0000m_Euro_NUT0_S197901010000_E202312312300_INS_TIM_01h_NA-_noc_org_NA_NA---_NA---_NA---.csv"
wind_speed_file = data_path + "H_ERA5_ECMW_T639_WS-_0010m_Euro_NUT0_S197901010000_E202312312300_INS_TIM_01h_NA-_noc_org_NA_NA---_NA---_NA---.csv"
start_date = '2021-01-01'
end_date = '2023-12-31'


def preprocess_data():
    temperature_df = load_and_process_weather_data(temperature_file, '_temperature', start_date, end_date, ['Date', 'DE', 'SE'])
    radiation_df = load_and_process_weather_data(radiation_file, '_radiation', start_date, end_date, ['Date', 'DE', 'SE'])
    wind_speed_df = load_and_process_weather_data(wind_speed_file, '_wind_speed', start_date, end_date, ['Date', 'DE', 'SE'])
    # Convert K to C for temperature data only
    temperature_df -= 273.15
    weather_data = temperature_df.join([radiation_df, wind_speed_df])
    print(weather_data.head)
    moer_data = load_moer_data()
    final_data = combine_data(weather_data, moer_data)
    print(final_data.head)
    return final_data


def load_and_process_weather_data(file_path, suffix, start_date, end_date, columns_of_interest):
    weather_data = pd.read_csv(file_path, skiprows=52)
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    weather_data.set_index('Date', inplace=True)
    weather_data = weather_data.loc[start_date:end_date]
    weather_data = weather_data.resample('H').ffill(limit=1).interpolate(method='linear')
    weather_data = weather_data[columns_of_interest].add_suffix(suffix)
    return weather_data


def load_moer_data():
    moer_data = pd.read_csv(os.path.join(data_path, 'moer_de_se.csv'))
    return moer_data


def combine_data(weather_data, moer_data):
    # Combine all data into a single DataFrame
    final_data = pd.DataFrame()
    weather_data.rename(columns={'Date': 'date'}, inplace=True)
    # weather_data['date'] = pd.to_datetime(weather_data['date'])
    # moer_data['date'] = pd.to_datetime(moer_data['date'])
    # weather_data.set_index(['date', 'country'], inplace=True)
    # moer_data.set_index(['date', 'country'], inplace=True)

    for country in ['DE', 'SE']:
        combined_df = moer_data[moer_data['country'] == country]
        for data_type in ['temperature', 'radiation', 'wind_speed']:
            weather_df = weather_data[f"{country}_{data_type}"]
            combined_df = combined_df.join(weather_df, how='left')
        final_data = pd.concat([final_data, combined_df.reset_index()], ignore_index=True)
    final_data['temperature'] = np.where(final_data['country'] == 'DE', final_data['DE_temperature'],
                                         final_data['SE_temperature'])
    final_data['radiation'] = np.where(final_data['country'] == 'DE', final_data['DE_radiation'],
                                       final_data['SE_radiation'])
    final_data['wind_speed'] = np.where(final_data['country'] == 'DE', final_data['DE_wind_speed'],
                                        final_data['SE_wind_speed'])
    final_data.drop(
        ['DE_temperature', 'DE_radiation', 'DE_wind_speed', 'SE_temperature', 'SE_radiation', 'SE_wind_speed'], axis=1,
        inplace=True)
    final_data.sort_values(by='date', inplace=True)
    final_data = final_data[['date', 'country', 'temperature', 'radiation', 'wind_speed', 'moer']]
    return final_data
