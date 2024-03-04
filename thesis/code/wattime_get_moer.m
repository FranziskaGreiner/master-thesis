# Define region and month
region = "DE"
start_date = datetime(2022, 10, 1)
end_date = datetime(2022, 11, 1)

df_name = f'moer_{region}_{start_date.year}_{start_date.month}'
dataframes = {}
dataframes[df_name] = pd.DataFrame()

# Define url and header with token from login
url = "https://api.watttime.org/v3/forecast/historical"
TOKEN = "my_watttime_login_token"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Get data from API
while start_date < end_date:
    end = start_date + timedelta(days=1)
    params = {
        "region": region,
        "start": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "end": end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "signal_type": "co2_moer",
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    if not data['data']:
        continue
    daily_data = pd.DataFrame(data['data'][0]['forecast'])
    dataframes[df_name] = pd.concat([dataframes[df_name], daily_data], ignore_index=True)

    start_date = end

# Resample data to one hour
dataframes[df_name]['point_time'] = pd.to_datetime(dataframes[df_name]['point_time'])
dataframes[df_name] = dataframes[df_name].resample('1H', on='point_time').mean().reset_index()

# Rename columns
dataframes[df_name].rename(columns={'point_time': 'date', 'value': 'moer'}, inplace=True)
dataframes[df_name]['date'] = pd.to_datetime(dataframes[df_name]['date'])

# Convert from lbs/MWh to g/kWh and delete timezone information
dataframes[df_name]['moer'] = dataframes[df_name]['moer'] * 453.592 / 1000
dataframes[df_name]['date'] = dataframes[df_name]['date'].dt.tz_localize(None)

# Save file
csv_file_path = f'/content/drive/My Drive/data_collection/WattTime/{df_name}.csv'
dataframes[df_name].to_csv(csv_file_path, index=False)
