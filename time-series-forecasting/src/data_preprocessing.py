import numpy as np
import pandas as pd
import glob
import os
from src.config import get_general_config

data_path = get_general_config()["data_path"]
temperature_file = data_path + "P_CMI5_CCMP_CM20_TA-_0002m_Euro_NUT0_S195001010000_E210012312100_INS_TIM_03h_NA-_cdf_org_01_RCP85_NA---_NA---.csv"
radiation_file = data_path + "P_CMI5_CCMP_CM20_GHI_0000m_Euro_NUT0_S195001010130_E210012312230_INS_TIM_03h_NA-_cdf_org_01_RCP85_NA---_NA---.csv"
wind_speed_file = data_path + "P_CMI5_CCMP_CM20_WS-_0100m_Euro_NUT0_S195001010000_E210012312100_INS_TIM_03h_NA-_cdf_org_01_RCP85_NA---_NA---.csv"
start_date = '2021-01-01'
end_date = '2023-12-31'


def preprocess_data():
    temperature_df = load_and_process_weather_data(temperature_file, 'Date', '_temperature', start_date, end_date,
                                                   ['DE', 'SE'])
    radiation_df = load_and_process_weather_data(radiation_file, 'Date', '_radiation', start_date, end_date, ['DE', 'SE'])
    wind_speed_df = load_and_process_weather_data(wind_speed_file, 'Date', '_wind_speed', start_date, end_date,
                                                  ['DE', 'SE'])
    # Convert K to C for temperature data only
    temperature_df -= 273.15
    weather_data = temperature_df.join([radiation_df, wind_speed_df])
    moer_data = load_and_process_moer_data()
    final_data = combine_data(weather_data, moer_data)
    return final_data


def load_and_process_weather_data(file_path, date_column, suffix, start_date, end_date, columns_of_interest):
    weather_data = pd.read_csv(file_path, skiprows=51)
    weather_data[date_column] = pd.to_datetime(weather_data[date_column])
    weather_data.set_index(date_column, inplace=True)
    weather_data = weather_data.loc[start_date:end_date]
    weather_data = weather_data.resample('H').ffill(limit=1).interpolate(method='linear')
    weather_data = weather_data[columns_of_interest].add_suffix(suffix)
    return weather_data


def load_and_process_moer_data():
    moer_de, moer_se = pd.DataFrame(), pd.DataFrame()
    moer_files_de = glob.glob(os.path.join(data_path, 'co2_moer_de_hourly_*.csv'))
    moer_files_se = glob.glob(os.path.join(data_path, 'co2_moer_se_hourly_*.csv'))

    for file in moer_files_de:
        df = pd.read_csv(file)
        df['country'] = 'DE'
        moer_de = pd.concat([moer_de, df], ignore_index=True)

    for file in moer_files_se:
        df = pd.read_csv(file)
        df['country'] = 'SE'
        moer_se = pd.concat([moer_se, df], ignore_index=True)

    moer_data = pd.concat([moer_de, moer_se], ignore_index=True)
    moer_data['date'] = pd.to_datetime(moer_data['date'])
    moer_data.rename(columns={'moer_value': 'moer'}, inplace=True)
    moer_data['date'] = moer_data['date'].dt.tz_localize(None)
    return moer_data


def combine_data(weather_data, moer_data):
    # Combine all data into a single DataFrame
    final_data = pd.DataFrame()
    for country in ['DE', 'SE']:
        combined_df = moer_data[moer_data['country'] == country].set_index('date')
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
