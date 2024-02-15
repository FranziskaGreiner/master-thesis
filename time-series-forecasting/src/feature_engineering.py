import pandas as pd
import holidays
import pytz
from datetime import datetime
from src.config import get_general_config

preprocessed_data_file_name = get_general_config()["preprocessed_data_file_name"]
data_path = get_general_config()["data_path"]


def add_features(final_data):
    print(final_data.head)
    final_data['date'] = final_data.apply(apply_timezone, axis=1)
    final_data.sort_values(by='date', inplace=True)
    final_data = final_data[['date', 'country', 'temperature', 'radiation', 'wind_speed', 'moer']]
    final_data['date'] = pd.to_datetime(final_data['date'])
    start_dates = {'DE': final_data[final_data['country'] == 'DE']['date'].min(),
                   'SE': final_data[final_data['country'] == 'SE']['date'].min()}
    final_data['time_idx'] = final_data.apply(lambda row: calculate_time_idx(row, start_dates), axis=1)
    final_data['is_holiday'] = final_data.apply(lambda row: is_holiday(row['date'], row['country']), axis=1)
    final_data['season'] = final_data['date'].apply(get_season)
    final_data['day_of_week'] = final_data['date'].dt.day_name()
    final_data['season'] = final_data['season'].astype('category')
    final_data['day_of_week'] = final_data['day_of_week'].astype('category')
    final_data['country'] = final_data['country'].astype('category')
    print(final_data.head)
    final_data.to_csv(f'{data_path}{preprocessed_data_file_name}', index=False)
    return final_data


def apply_timezone(row):
    germany_tz = pytz.timezone('Europe/Berlin')
    sweden_tz = pytz.timezone('Europe/Stockholm')

    if row['country'] == 'DE':
        return row['date'].tz_localize('UTC').tz_convert(germany_tz).tz_localize(None)
    elif row['country'] == 'SE':
        return row['date'].tz_localize('UTC').tz_convert(sweden_tz).tz_localize(None)
    return row['date']


def calculate_time_idx(row, start_dates):
    country_start_date = start_dates[row['country']]
    return (row['date'] - country_start_date).total_seconds() // 3600


def is_holiday(date, country_code):
    try:
        country_holidays = holidays.country_holidays(country_code)
        return int(date in country_holidays)
    except KeyError:
        return 0


def get_season(date):
    year = date.year
    spring_start = datetime(year, 3, 20)
    summer_start = datetime(year, 6, 21)
    autumn_start = datetime(year, 9, 23)
    winter_start = datetime(year, 12, 21)
    if spring_start <= date < summer_start:
        return 'spring'
    elif summer_start <= date < autumn_start:
        return 'summer'
    elif autumn_start <= date < winter_start:
        return 'autumn'
    else:
        return 'winter'



