import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import colorspacious
import webbrowser
import json
import requests_cache
from retry_requests import retry
from scipy.ndimage import convolve
from os import getenv

class Path:
    def __init__(self, n_edge, n_inner):
        self.n_edge = n_edge
        self.n_inner = n_inner
        self.load_df()
        self.get_info()
        self.get_flat_info()
        self.color_mapping = get_color_mapping()

    def __call__(self, n_call=200):
        self.cloud_f = get_cloud(self.lat_f, self.lng_f, self.time, n_call=n_call)
        self.cloud = self.cloud_f.reshape(self.n, self.m)
        self.write_json()

    def load_df(self):
        self.df = pd.read_csv('locs.csv', index_col=0)
        self.df = self.df[5:39].reset_index(drop=True)
        self.df = convert_to_decimal(self)

    def get_info(self):
        """Returns information on eciplse path, grid of lat and lng."""
        df = get_boundaries(self.df, self.n_edge)
        self.lat, self.lng = get_inner(df, self.n_inner)
        self.n, self.m = self.lat.shape
        self.lat, self.lng = self.lat.round(2), self.lng.round(2)
        self.time = df.Time.map(convert_to_timestamp).values

    def get_flat_info(self):
        # A degree is about 111 km so 0.01 degrees is about 1.1 km
        self.lat_f = self.lat.flatten()
        self.lng_f = self.lng.flatten()
        self.time = np.repeat(self.time, self.n_inner+1)
        assert self.lat_f.shape == self.lng_f.shape == self.time.shape == (len(self.lat_f),)

    def write_json(self):
        kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
        self.cloud_m = convolve(self.cloud.astype(float), kernel, mode='constant', cval=0).astype(int)[:-1,:-1]
        top_left_n, top_right_n, bottom_left_n, bottom_right_n = get_corners(self.lng)
        top_left_a, top_right_a, bottom_left_a, bottom_right_a = get_corners(self.lat)
        features = []
        for i in range(self.n-1):
            for j in range(self.m-1):
                feature = {}
                feature["type"] = "Feature"
                coordinates = []
                coordinates.append([top_left_n[i, j], top_left_a[i, j]])
                coordinates.append([top_right_n[i, j], top_right_a[i, j]])
                coordinates.append([bottom_right_n[i, j], bottom_right_a[i, j]])
                coordinates.append([bottom_left_n[i, j], bottom_left_a[i, j]])
                coordinates.append([top_left_n[i, j], top_left_a[i, j]])
                feature["properties"] = {}
                cloud_cover = self.cloud_m[i, j]
                feature["properties"]["cloud_cover"] = str(cloud_cover)
                feature["properties"]["color"] = f"rgb{self.color_mapping[cloud_cover]}"
                feature["geometry"] = {}
                feature["geometry"]["type"] = "Polygon"
                feature["geometry"]["coordinates"] = [coordinates]
                features.append(feature)
        json_file = {}
        json_file["type"] = "FeatureCollection"
        json_file["features"] = features
        self.json_file = json_file
        file_path = getenv('CLOUDS_JSON_PATH')
        with open(file_path, 'w') as file:
            json.dump(json_file, file)

def get_corners(array):
    top_left = array[:-1, :-1]
    top_right = array[:-1, 1:]
    bottom_left = array[1:, :-1]
    bottom_right = array[1:, 1:]
    return top_left, top_right, bottom_left, bottom_right

def get_rgb_colors():
    colors = plt.cm.bwr(np.linspace(0, 1, 101))
    rgb_colors = (colors[:, :3] * 255).astype(int)
    return rgb_colors

def view_color_mapping():
    rgb_colors = get_rgb_colors()
    gradient_image = rgb_colors.reshape(1, 101, 3)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(gradient_image.astype(np.uint8), aspect='auto')
    ax.set_yticks([])
    ax.set_xticks([i for i in range(0, 101, 10)])
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 10)])
    ax.set_xlabel("Cloud Cover Percentage")
    plt.show()

def get_color_mapping():
    rgb_colors = get_rgb_colors()
    return {i: tuple(color) for i, color in enumerate(rgb_colors)}

def _archived_color_mapping(start_color=(137, 183, 249), end_color=(10, 10, 10)):
    # start color was initially start_color=(54, 136, 255)
    # we also considered (137, 183, 249),
    gradient_cielab = [_interpolate_color_cielab(start_color, end_color, i / 100.0) for i in range(101)]
    color_mapping = {i: rgb for i, rgb in enumerate(gradient_cielab)}
    return color_mapping

def _interpolate_color_cielab(color1, color2, fraction):
    # Convert from RGB to CIELAB
    lab1 = colorspacious.cspace_convert(color1, "sRGB255", "CIELab")
    lab2 = colorspacious.cspace_convert(color2, "sRGB255", "CIELab")
    # Interpolate in CIELAB space
    lab_interp = lab1 + (lab2 - lab1) * fraction
    # Convert the interpolated color back to RGB
    rgb_interp = colorspacious.cspace_convert(lab_interp, "CIELab", "sRGB255")
    return tuple(max(0, min(255, round(x))) for x in rgb_interp)

def get_cloud(lat, lng, time, n_call=260):
    n_points = len(lat)
    cloud = np.zeros(n_points)
    for i in range(0, n_points, n_call):
        # sleep(0)
        lat_call, lng_call, time_call = lat[i:i+n_call], lng[i:i+n_call], time[i:i+n_call]
        responses = get_responses(lat_call, lng_call)
        cloud_call = get_cloud_call(time_call, responses)
        cloud[i:i+n_call] = cloud_call
    return cloud

def get_cloud_call(time_call, responses):
    cloud_call = np.zeros(len(responses))
    for response_num in range(len(responses)):
        cloud_call[response_num] = get_cloud_point(response_num, time_call, responses)
    return cloud_call

def get_cloud_point(response_num, time_call, responses):
    hourly = get_hourly(response_num, responses)
    time_point = get_time_point(time_call, response_num)
    return hourly.loc[time_point]

def get_time_point(time_call, response_num):
    time_point = time_call[response_num]
    time_point = pd.Timestamp(time_point, tz='UTC').ceil("H")
    return time_point

def get_hourly(response_num, responses):
    response = responses[response_num]
    hourly = response.Hourly()
    hourly_cloud_cover = hourly.Variables(0).ValuesAsNumpy()
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data = pd.Series(hourly_data["cloud_cover"], index=hourly_data["date"])
    return hourly_data

def get_responses(lat_call, lng_call, n_days=3):
    url = "https://customer-api.open-meteo.com/v1/forecast"
    api_key = getenv('OPENMETEO_API_KEY')
    params = {
        "latitude": lat_call,
        "longitude": lng_call,
        "hourly": "cloud_cover",
        "timezone": "GMT",
        "forecast_days": n_days,
        "apikey": api_key
        }
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    responses = openmeteo.weather_api(url, params=params)
    return responses

def convert_to_timestamp(time_str):
    # Base date string for April 8, 2024 in UTC
    base_date = "2024-04-08"
    # Combine the base date with the time string
    datetime_str = f"{base_date} {time_str}"
    # Convert to pandas timestamp
    timestamp = pd.to_datetime(datetime_str, utc=True)
    return timestamp

def get_inner(df, n):
    """Returns inner grid indexed by lat and long, where each row bisects the path."""
    lat = np.linspace(df.North_Latitude, df.South_Latitude, n + 1, endpoint=True, axis=1)
    long = np.linspace(df.North_Longitude, df.South_Longitude, n + 1, endpoint=True, axis=1)
    return lat, long

def get_boundaries(df, n):
    """Returns pd DataFrame of Time and eclipse boundaries."""
    boundary_cols = ['North_Latitude', 'North_Longitude', 'South_Latitude', 'South_Longitude']
    dfn = pd.DataFrame()
    dfn["Time"] = get_time(df.Time.values, n)
    for col in boundary_cols:
        dfn[col] = interpolate_boundary(df[col].values, n)
    assert len(dfn) == get_len(df.shape[0], n)
    return dfn

def dms_to_dd(dms_str):
    parts = dms_str.split('°')
    degrees = int(parts[0])
    minutes, direction = parts[1].split("'")
    minutes = float(minutes)
    dd = degrees + (minutes / 60)
    if direction in ['S', 'W']:
        dd *= -1
    return dd

def convert_to_decimal(df):
    loc_cols = ['North_Latitude', 'North_Longitude', 'South_Latitude',
                'South_Longitude', 'Centr_Latitude', 'Centr_Longitude']
    for col in loc_cols:
        df[col] = df[col].map(lambda x: dms_to_dd(x))
    return df

def get_len(size, scaling):
    return size  * scaling - (scaling - 1)

def interpolate_boundary(array, n):
    ans_len = get_len(len(array), n)
    ans = np.zeros(ans_len)
    for i in range(len(array) - 1):
        ans[i * n:(i + 1) * n] = np.linspace(array[i], array[i + 1], n, endpoint=False)
    ans[-1] = array[-1]
    return ans

def get_time(time, n):
    return np.repeat(time, n)[:n*len(time)-n+1]

def test_time():
    n = 3
    time = np.array([1, 2, 3, 4, 9])
    assert len(get_time(time, n)) == get_len(time.size, n)

def get_boudary_stats(path):
    stats = {"Lat edge end max": np.abs(path.lat[-2, :] - path.lat[-1, :]).max().round(4),
    "Lat inner end max": np.abs(path.lat[:, -2] - path.lat[:, -1]).max().round(4),
    "Lng edge end max": np.abs(path.lng[-2, :] - path.lng[-1, :]).max().round(4),
    "Lng inner end max": np.abs(path.lng[:, -2] - path.lng[:, -1]).max().round(4),
    "Lat edge start max": np.abs(path.lat[0, :] - path.lat[1, :]).max().round(4),
    "Lat inner start max": np.abs(path.lat[:, 0] - path.lat[:, 1]).max().round(4),
    "Lng edge start max": np.abs(path.lng[0, :] - path.lng[1, :]).max().round(4),
    "Lng inner start max": np.abs(path.lng[:, 0] - path.lng[:, 1]).max().round(4)
    }
    return stats

def open_google_maps(lat, lng):
    maps_url = f"https://www.google.com/maps?q={lat},{lng}"
    webbrowser.open(maps_url)
