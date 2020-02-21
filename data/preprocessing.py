# Third-party.
import pandas as pd
from sklearn import preprocessing as pp


def get_wind_energy():
    wind_energy = pd.read_csv(
        filepath_or_buffer='/Users/oliver/Documents/networks/data/static_data/wind_energy.csv',
        sep=',',
        parse_dates=[['date', 'time']],
        index_col='date_time',
        dtype={
            'p_1': float,
            'p_2': float,
            'p_3': float,
            'p_t': float,
        },
        engine='c',
        infer_datetime_format=True,
        cache_dates=True,
    )
    # up-sample data to match 10min intervals.
    wind_upsampled = wind_energy.resample('10T').mean()
    wind_upsampled_interpolated = wind_upsampled.interpolate(method='spline', order=2)

    we_power_totals = []

    # wind farm readings need to be added up and converted to positive values before being used.
    for row in wind_upsampled_interpolated.iterrows():
        total_power = [abs(row[1].p_1) + abs(row[1].p_2) + abs(row[1].p_3)]
        we_power_totals.append(total_power)

    return we_power_totals


def get_weather_10_min_interval():
    weather_data_10 = pd.read_csv(
        filepath_or_buffer='/Users/oliver/Documents/networks/data/static_data/weather_data_10.csv',
        sep=',',
        parse_dates=[['date', 'time']],
        index_col='date_time',
        dtype={
            'airtemp': float,
            'winddirection': float,
            'windspeed': float,
            'sunshineduration': float,
            'airpressure': float,
            'precipitation': float,
        },
        engine='c',
        infer_datetime_format=True,
        cache_dates=True,
    )
    # standardize data.
    w10_scaled_values = pp.scale(weather_data_10.values)

    return w10_scaled_values


