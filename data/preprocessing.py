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
    # upsample data to match 10min intervals.
    wind_upsampled = wind_energy.resample('10T').mean()
    wind_upsampled_interpolated = wind_upsampled.interpolate(method='spline', order=2)

    we_power_totals = []

    # wind farm readings need to be added up and converted to positive values before being used.
    for row in wind_upsampled_interpolated.iterrows():
        total_power = [abs(row[1].p_1) + abs(row[1].p_2) + abs(row[1].p_3)]
        we_power_totals.append(total_power)

    return we_power_totals


def get_solar_energy():
    solar_energy = pd.read_csv(
        filepath_or_buffer='static_data/solar_energy.csv',
        sep=',',
        parse_dates=[['date', 'time']],
        dtype={'p_s': float},
        engine='c',
        infer_datetime_format=True,
        cache_dates=True,
    )
    # no need to upsample as already in 10min intervals.

    # get values as an array of arrays excluding the times and dates, suitable input for neural networks.
    se_power = []

    for row in solar_energy.iterrows():
        solar_power = row[1].p_s
        se_power.append(solar_power)

    # standardize data.
    se_power_scaled = pp.scale(se_power)

    return se_power_scaled.tolist()


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


def get_weather_60_min_interval():
    weather_data_60 = pd.read_csv(
        filepath_or_buffer='static_data/weather_data_60.csv',
        sep=',',
        parse_dates=[['date', 'time']],
        index_col='date_time',
        dtype={
            'cloudcoverage': int,
            'winddirection': float,
            'windspeed': float,
            'airtemp': float,
            'airpressure': float,
            'sunshineduration': float,
            'precipitation': float,
        },
        engine='c',
        infer_datetime_format=True,
        cache_dates=True,
    )
    weather_60_upsampled = weather_data_60.resample('10T').mean()
    weather_60_upsampled_interpolated = weather_60_upsampled.interpolate(method='spline', order=2)

    # standardize data.
    w60_scaled_values = pp.scale(weather_60_upsampled_interpolated.values)

    return w60_scaled_values


def get_weather_forecast_data():
    weather_forecast = pd.read_csv(
        filepath_or_buffer='static_data/weather_forecast.csv',
        sep=',',
        parse_dates=[['datevalid', 'timevalid']],
        index_col='datevalid_timevalid',
        dtype={
            'temp': float,
            'dewpoint': float,
            'windspeed': float,
            'gustspeed': float,
            'airpressure': float,
            'precipprob': float,
            'cloudcoverage': int,
            'solarirradiance': float,
            'winddirection': float,
            'airhumidity': float,
            'airdensity': float,
        },
        engine='c',
        infer_datetime_format=True,
        cache_dates=True,
    )
    weather_forecast_upsampled = weather_forecast.resample('10T').mean()
    weather_forecast_upsampled_interpolated = weather_forecast_upsampled.interpolate(method='spline', order=2)

    # standardize data.
    wf_scaled_values = pp.scale(weather_forecast_upsampled_interpolated.values)

    return wf_scaled_values

