'''
Post Processing for AdMos Live workflow

'''

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy as sp
import pywt
import streamlit as st
from streamlit_plotly_events import plotly_events
import math
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy import stats
import datetime
from zoneinfo import ZoneInfo
import requests




st.set_page_config(layout="wide")

image, title = st.columns([1,9])

with image: 
    st.image('https://images.squarespace-cdn.com/content/v1/5f63780dbc8d16716cca706a/1604523297465-6BAIW9AOVGRI7PARBCH3/rowing-canada-new-logo.jpg')
with title:
    st.title('GPS Post Processing')



def lowpass(signal, highcut, frequency):
    '''
    Apply a low-pass filter using Butterworth design

    Inputs;
    signal = array-like input
    high-cut = desired cutoff frequency
    frequency = data sample rate in Hz

    Returns;
    filtered signal
    '''
    order = 2
    nyq = 0.5 * frequency
    highcut = highcut / nyq #normalize cutoff frequency
    b, a = sp.signal.butter(order, [highcut], 'lowpass', analog=False)
    y = sp.signal.filtfilt(b, a, signal, axis=0)
    return y

def highpass(signal, lowcut, frequency):
    '''
    Apply a low-pass filter using Butterworth design

    Inputs;
    signal = array-like input
    high-cut = desired cutoff frequency
    frequency = data sample rate in Hz

    Returns;
    filtered signal
    '''
    order = 2
    nyq = 0.5 * frequency
    lowcut = nyq/lowcut #normalize cutoff frequency
    b, a = sp.signal.butter(order, [lowcut], 'highpass', analog=False)
    y = sp.signal.filtfilt(b, a, signal, axis=0)
    return y

def swt_coeffs(data, wavelet, levels, coeffs_out):

    '''
    Perform stationary wavelet transformation on input signal

    PyWavelet follows the “algorithm a-trous” and requires that the signal length along the transformed axis
    be a multiple of 2**level. To ensure that the input signal meets this condition, we use numpy.pad to pad
    the signal symmetrically, then crop the padding off of the returned coefficients.


    Inputs;
    data = array-like input
    wavelet = string of specific wavelet function eg: 'bior3.1'
    levels = int levels of decomposition
    coeffs_out = list of desired coefficient levels eg: [0,2,9]

    Returns:
    dataframe containing approximation (low freq) and detail (high freq) coefficients for each specified level from coeffs_out
    coefficients are returned with the format cAx and cDx where A is approximation, D is detail and x is level
    '''
    # Calculate the necessary padding size
    target_length = np.ceil(len(data) / (2**levels)) * (2**levels)
    padding_size = int(target_length - len(data))

    # Pad the data symmetrically so that is a multiple of 2^levels
    data_padded = np.pad(data, (padding_size // 2, padding_size - padding_size // 2), mode='symmetric')

    # Perform undecimated wavelet transform
    coeffs = pywt.swt(data_padded, wavelet=wavelet, level=levels)

    # Create a dictionary to store the desired coefficients
    result = {}

    for level in coeffs_out:
        # Get the approximation and detail coefficients for the desired level
        approximation, detail = coeffs[level]

        # Crop the coefficients to the original data size
        approximation_cropped = approximation[padding_size // 2 : - (padding_size - padding_size // 2)]
        detail_cropped = detail[padding_size // 2 : - (padding_size - padding_size // 2)]

        # Store the coefficients in the result dictionary
        result[f'cA{level}'] = approximation_cropped
        result[f'cD{level}'] = detail_cropped

    # Convert the result dictionary to a DataFrame
    df_result = pd.DataFrame(result)

    return df_result

def gps_stroke_stack(velocity, indices):
    """
    Analyzes GPS data for each stroke.

    Stacking the strokes
    -Stroke number to be the index
    -Velocity values as columns

    """
    stroke_vels = []
    for i in range(0,len(indices)-1): 
        stroke_vel = list(velocity.iloc[indices[i]:indices[i+1]])
        stroke_vels.append(stroke_vel)

    return stroke_vels

def cumulative_distance_array(latitudes, longitudes):
    """
    Given arrays/lists of latitudes and longitudes (in degrees),
    return an array of the cumulative distance (in meters) up to each point.
    
    For example, if you have n points, the result will be an array
    of length n where:
        distances[0] = 0
        distances[i] = distances[i-1] + distance from point i-1 to i (for i >= 1)
    
    Parameters
    ----------
    latitudes : list or array-like
        Latitude values in degrees.
    longitudes : list or array-like
        Longitude values in degrees.

    Returns
    -------
    distances : list
        Cumulative distance for each point in meters.
    """
    if len(latitudes) != len(longitudes):
        raise ValueError("Latitudes and longitudes must have the same length.")

    R = 6371000  # Approximate radius of Earth in meters
    n = len(latitudes)
    
    # Initialize array with 0 as the first distance
    distances = [0] * n  
    
    for i in range(1, n):
        lat1 = math.radians(latitudes[i - 1])
        lon1 = math.radians(longitudes[i - 1])
        lat2 = math.radians(latitudes[i])
        lon2 = math.radians(longitudes[i])
        
        # Differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = (math.sin(dlat / 2)**2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        segment_distance = R * c  # Distance in meters
        
        # Add this segment distance to the previous total
        distances[i] = distances[i - 1] + segment_distance
    
    return distances

def speed_to_split(speed):
    seconds = 500/speed
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"

def sec_to_split(seconds):
    
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"

def speed_to_split_dist(speed, distance):
    seconds = distance/speed
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"

def boxplot_upper_whisker(data):
    """
    Returns the maximum (upper) box plot whisker value for 'data',
    defined as the largest data point <= Q3 + 1.5 * IQR.
    
    Parameters:
    -----------
    data : list or array-like
        A sequence of numerical data.
    
    Returns:
    --------
    float
        The maximum whisker value for the box plot, i.e. the largest
        data point within 1.5 * IQR above Q3.
    """
    # Convert to a NumPy array for convenience
    arr = np.array(data, dtype=float)
    
    # Calculate Q1 and Q3
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 85)
    
    # Interquartile Range
    iqr = q3 - q1
    
    # Upper bound for the boxplot whisker
    upper_bound = q3 + 1.5 * iqr
    
    # The whisker is the largest data point at or below upper_bound
    # That might be the upper_bound itself (if no data exceeds it),
    # but usually it's the largest data point that is <= upper_bound.
    return arr[arr <= upper_bound].max()


#fetch data from API
start_enter, ende_enter = st.columns(2)
with start_enter:
    start_date_time = st.text_input('Enter Starting Date and Time')
with ende_enter:
    end_date_time = st.text_input('Enter Ending Date and Time')
device_select = st.selectbox('Select Device to Pull', ['22621',
'22662', 
'22953', 
'22993', 
'22999'])


st.write("Example Date and Time Format: 2025-04-09T10:15:00")


def get_preprocessed_data(
    device_id,
    access_token,
    start_time,
    stop_time,
    page=1,
    limit=25000
):
    """
    Retrieve a *single page* of data for a specific device within a time range,
    using the 'page' parameter for pagination (instead of offset).
    """
    url = f"https://api.insiders.live/v1/devices/{device_id}/preprocessed/"
    
    params = {
        "start": start_time,
        "stop": stop_time,
        "page": page,
        "limit": limit,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.json()

def get_all_preprocessed_data(device_id, access_token, start_time, stop_time):
    """
    Uses `get_preprocessed_data(...)` to keep pulling data in 25,000-row pages
    until the total count is reached.
    """
    page = 1
    limit = 25000
    all_results = []
    total_count = None

    while True:
        # Fetch this page
        data_page = get_preprocessed_data(device_id, access_token, start_time, stop_time,
                                          page=page, limit=limit)
        
        # The first page usually includes the total count; grab it if we don't have it yet
        if total_count is None:
            total_count = data_page.get('count', 0)
        
        # Extract the page's results
        page_results = data_page.get('results', [])
        all_results.extend(page_results)
        
        # If we got fewer than 'limit' rows OR we've reached/exceeded total_count, we can stop
        if len(page_results) < limit or len(all_results) >= total_count:
            break
        
        # Otherwise, move on to the next page
        page += 1

    # Convert all rows to a DataFrame
    df = pd.DataFrame(all_results)
    return df

def convert_iso8601_to_utc_string(iso_time_str: str) -> str:
    """
    Convert an ISO 8601 datetime string to an ISO 8601 string in UTC ("Z" at the end).
    
    1) If iso_time_str includes an offset (e.g. "-08:00"), Python will parse it as a timezone-aware datetime.
    2) If iso_time_str is naive (no offset), assume it is in "America/Los_Angeles" (PST/PDT).
    3) Finally, convert to UTC and produce a string in the format "YYYY-MM-DDTHH:MM:SSZ".
    """
    dt = datetime.datetime.fromisoformat(iso_time_str)
    
    # If there's no timezone info, assume America/Los_Angeles
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        
    # Convert to UTC
    dt_utc = dt.astimezone(datetime.timezone.utc)
    
    # Return ISO 8601 string with a trailing 'Z' instead of '+00:00'
    return dt_utc.isoformat().replace("+00:00", "Z")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_pulled' not in st.session_state:
    st.session_state.data_pulled = False

pull = st.sidebar.checkbox('Pull Data')
if st.sidebar.button("Reset Data"):
    st.session_state.data = None
    st.session_state.data_pulled = False


if pull and not st.session_state.data_pulled:


    client_id = 'M9d3J3axpMU9z9xMfaqNtOB4BdxmsyPMK2v63yBC'
    client_secret = 'POh9pZ1djjNOtS8rX9FzTYHAv3ARYIvaht9pfXlKPc3axcTaCPoxYehS3OVOGhRUn9ahaujugmftpajWC7zAW0LoxVEBMhhEIg86D4Yp5g05KfT9SLGSrin6oyd0SNnd'

    #Your device manager account info here
    username = "dgeneau@csipacific.ca"
    password = "Joegeneau!1959"


    data_login = {
    'grant_type': 'password',
    'username': username,
    'password': password
    }

    response = requests.post('https://api.asi.swiss/oauth2/token/',
                            data=data_login,
                            verify=False,
                            allow_redirects=False,
                            auth=(client_id, client_secret)).json()

    access_token = response['access_token']


    device_id = device_select


    ACCESS_TOKEN = access_token
    START_TIME = convert_iso8601_to_utc_string(start_date_time) # Example ISO 8601 format
    STOP_TIME = convert_iso8601_to_utc_string(end_date_time)  # Example ISO 8601 format




    try:
        df = get_all_preprocessed_data(device_id, ACCESS_TOKEN, START_TIME, STOP_TIME)
        
        if not df.empty:
            # Process timestamps
            timestamp_seconds = df['gnss.timestamp'] / 1000
            local_time = [
                datetime.datetime.utcfromtimestamp(ts)
                .replace(tzinfo=datetime.timezone.utc)
                .astimezone(ZoneInfo("America/Los_Angeles"))
                for ts in timestamp_seconds
            ]
            df['gnss.timestampPC'] = local_time

            # Save to session_state
            st.session_state.data = df
            st.session_state.data_pulled = True  
            st.success("Data pulled and stored successfully.")

    except Exception as e:
        st.error(f"Error during data pull: {e}")
        st.session_state.data_pulled = False

# Use the data later in your app
if st.session_state.data is not None:
    
    data = st.session_state.data
else:
    st.warning("Please pull data to continue.")
    st.stop()



        

#########################################################################
_='''
folder_list = glob.glob('/Users/danielgeneau/Library/CloudStorage/OneDrive-CanadianSportInstitutePacific/Dash Sandbox/**/', recursive=True)
folder_list = glob.glob('/Users/danielgeneau/Library/CloudStorage/OneDrive-SharedLibraries-RowingCanadaAviron/HP - Staff - SSSM/General/Biomechanics/Sensor Project/Insiders/**/', recursive=True)
folder_sel = st.selectbox('Select Folder for Analysis', folder_list)


if folder_sel is not None:
    data_list = glob.glob(f'{folder_sel}/*.csv')
    data_sel = st.selectbox('Select Data for Analysis', data_list)
'''


prog_dict = {
                    "M8+":"6.269592476",
                    "M4-":"5.899705015",
                    "M2-":"5.376344086",
                    "M4x":"6.024096386",
                    "M2x":"5.555555556",
                    "M1x":"5.115089514",
                    "W8+":"5.66572238",
                    "W4-":"5.347593583",
                    "W2-":"4.901960784",
                    "W4x":"5.464480874",
                    "W2x":"5.037783375",
                    "W1x":"4.672897196",
                }

prog = st.sidebar.selectbox('Select Boat Classs', prog_dict)

prog_vel = prog_dict[prog]
#Upsampling
factor = 3
original_length = len(data)

if 'gnss.speed' in data.columns: 
    data['speed'] = data['gnss.speed']
    data['latitude'] = data['gnss.latitude']
    data['longitude'] = data['gnss.longitude']
    data['altitude'] = data['gnss.altitude']
    data['timestamp'] = data['gnss.timestamp']
   

#data['speed'] = lowpass(data['speed'], 1.5, 10)

if original_length > 2:
    old_indices = np.arange(original_length)
    new_indices = np.linspace(0, original_length - 1, factor * original_length)

    # Create a cubic spline interpolation function for 'speed'
    spline_speed = interp1d(old_indices, data['speed'], kind='quadratic')
    floor_indices = np.floor(new_indices).astype(int)
    upsample_timestamp = data['timestamp'].iloc[floor_indices].values

    # Evaluate the spline at the new indices
    upsample_speed = spline_speed(new_indices)
    upsample_lat = np.interp(new_indices, old_indices, data['latitude'])
    upsample_long = np.interp(new_indices, old_indices, data['longitude'])

    # Compute the approximate acceleration
    upsample_accel = np.diff(upsample_speed) / (1 / (10 * factor))





accel  = np.diff(data['speed'])/(1/10)
accel_neg = accel*-1
accel_neg = lowpass(accel_neg, 2, 5)
accel_plot = lowpass(accel, 2, 10)
speed_plot = lowpass(data['speed'], 2, 10)
up_frame = pd.DataFrame()
up_frame['speed'] = upsample_speed
up_frame['accel'] = np.insert(upsample_accel,0,0)
up_frame['lat'] = upsample_lat
up_frame['long'] = upsample_long
up_frame['timestamp'] = upsample_timestamp




coeff_df = swt_coeffs(up_frame['accel'], 'bior5.5', 7, [0,1,2,3,4,5,6]) 



data['distance'] = cumulative_distance_array(data['latitude'], data['longitude'])
up_frame['distance'] = cumulative_distance_array(up_frame['lat'], up_frame['long'])

combo_wave = coeff_df['cA4']*-1 + coeff_df['cA5']*-1 + coeff_df['cA6']*-1 +coeff_df['cD2']+coeff_df['cD1']


peaks,_ = find_peaks(combo_wave, height= 15, distance = factor*10,  width = [0, 40*factor])

session_time = (data['timestamp'].iloc[-1]-data['timestamp'].iloc[int(peaks[0]/3)])/1000
session_distance = (np.mean(data['speed'].iloc[int(peaks[0]/3):-1]))*(session_time)
sess_dist, sess_time, spacer = st.columns([2,2,5])

with sess_dist:
    st.metric('Session Distance (km)', round(session_distance/1000, 2))
with sess_time:
    st.metric('Session Time',sec_to_split(session_time) )
    


stroke_data = pd.DataFrame()

def rolling_average_five(arr):
    """
    Compute a rolling average over a 5-sample window.
    Replace the first 5 values in the result with the first computed average.
    """
    # Make a copy so original data isn't overwritten
    result = np.copy(arr).astype(float)
    n = 7
    
    # We'll store the rolling averages in a new array
    rolling_avgs = np.zeros(len(arr))
    
    # Compute the rolling average for indices >= n
    # (i.e., once we have at least 5 values)
    for i in range(n, len(arr) + 1):
        avg = np.mean(arr[i-n:i])
        rolling_avgs[i-1] = avg
    
    # Fill any unused positions at the end (if any) with the last average
    # (especially relevant if your array length isn't a multiple of 5)
    for i in range(len(arr)):
        if i >= len(arr) - (n - 1):
            rolling_avgs[i] = rolling_avgs[len(arr) - n]
    
    # Replace the first 5 values with the first computed average
    first_avg = rolling_avgs[n-1]
    rolling_avgs[:n] = first_avg
    
    return rolling_avgs


rates = 60/(np.diff(peaks)/(10*factor))


# Identify outliers using IQR method
def detect_outliers_iqr(data, factor=1.5):
    q1 = np.percentile(data, 10)
    q3 = np.percentile(data, 90)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return np.where((data < lower_bound) | (data > upper_bound))[0], lower_bound, upper_bound

# Identify outliers using Z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return np.where(z_scores > threshold)[0]

# Apply Z-Score method to detect outliers
outliers = detect_outliers_zscore(rates, threshold=3)


data_local_replaced = rates.copy()
window_size = 10  # number of adjacent points to consider

for idx in outliers:
    # Define the window boundaries, handling edge cases
    start = max(0, idx - window_size // 2)
    end = min(len(rates), idx + window_size // 2 + 1)
    
    # Get non-outlier values in the window
    window_indices = np.arange(start, end)
    non_outlier_indices = window_indices[~np.isin(window_indices, outliers)]
    
    if len(non_outlier_indices) > 0:
        # Replace with median of non-outlier values in window
        data_local_replaced[idx] = np.median(rates[non_outlier_indices])
    else:
        # If all window values are outliers, use global median
        data_local_replaced[idx] = np.median(rates[~np.isin(np.arange(len(rates)), outliers)])

stroke_data['Rate'] = data_local_replaced




stroke_speeds = []
stroke_min_s = []
stroke_max_s = []
stroke_accels = []
stroke_dist = []
stroke_ts = []
for i in range(1,len(peaks)): 
    stroke_speed = np.mean(up_frame['speed'].iloc[peaks[i-1]:peaks[i]]) 
    stroke_min_speed = np.min(up_frame['speed'].iloc[peaks[i-1]:peaks[i]])
    stroke_max_speed = np.max(up_frame['speed'].iloc[peaks[i-1]:peaks[i]])
    stroke_ts.append(up_frame['timestamp'].iloc[peaks[i-1]])
    stroke_min_s.append(stroke_min_speed)
    stroke_max_s.append(stroke_max_speed)
    stroke_speeds.append(stroke_speed)

    stroke_accel = np.mean(up_frame['accel'].iloc[peaks[i-1]:peaks[i]])
    stroke_accels.append(stroke_accel)

    stroke_d= up_frame['distance'].iloc[peaks[i]]
    stroke_dist.append(stroke_d)


stroke_data['speed'] = stroke_speeds
stroke_data['min speed'] = stroke_min_s
stroke_data['max speed'] = stroke_max_s
stroke_data['accel'] = stroke_accels
stroke_data['eWPS'] = stroke_data['speed']**3/stroke_data['Rate']
stroke_data['onset_index'] = peaks[1:]
stroke_data['timestamp'] = stroke_ts
stroke_data['DPS'] = stroke_speeds*(np.diff(peaks)/(10*factor))
stroke_data['Distance'] = stroke_dist

stroke_data = stroke_data[stroke_data['Rate']>10].reset_index(drop = True)

stroke_data['SR Round'] = stroke_data['Rate'].astype(int)

#####################################
default_bins = "15, 18, 22, 26, 30, 34, 38, 41, 44, inf"
bin_input = st.sidebar.text_input("Enter bin edges (comma-separated):", default_bins)

try:
    # Parse bin edges from input
    bin_edges = [float(b.strip()) if b.strip().lower() != "inf" else np.inf for b in bin_input.split(",")]

    # Validate enough edges
    if len(bin_edges) < 2:
        st.error("Please enter at least two bin edges.")
    else:
        # Generate labels like '15-18', '19-22', ..., '44+'
        labels = []
        for i in range(len(bin_edges) - 1):
            left = int(bin_edges[i])
            right = bin_edges[i + 1]
            if right == np.inf:
                labels.append(f"{left}+")
            else:
                labels.append(f"{left}-{int(right)}")

        # Perform binning
        stroke_data['Rate_Bin'] = pd.cut(stroke_data['SR Round'], bins=bin_edges, labels=labels, right=True)
        
except Exception as e:
    st.error(f"Error parsing inputs or applying binning: {e}")



SR_fig = go.Figure()
SR_fig.add_trace(go.Scattergl(
    y = list(stroke_data['Rate']),
    x = list(stroke_data['onset_index']),
    mode = 'markers', 
    name = 'Stroke Rate (SPM)', 
    marker_color = 'blue'
))
SR_fig.add_trace(go.Scattergl(
    y = list(stroke_data['speed']),
    x = list(stroke_data['onset_index']), 
    mode = 'markers', 
    name = 'Speed (m/s)', 
    yaxis = 'y2'
))

SR_fig.add_trace(go.Scattergl(
    y = list(stroke_data['eWPS']), 
    x = list(stroke_data['onset_index']),
    mode = 'markers', 
    name = 'E Work Per Stroke', 
    yaxis = 'y2', 
    marker_color = 'green',
    opacity=0.5
))

SR_fig.add_trace(go.Scattergl(
    y = list(stroke_data['max speed']), 
    x = list(stroke_data['onset_index']),
    mode = 'markers', 
    name = 'Max Speed (m/s)', 
    yaxis = 'y2',  
    marker_color = 'red'
))

SR_fig.add_trace(go.Scattergl(
    y = list(stroke_data['min speed']), 
    x = list(stroke_data['onset_index']),
    mode = 'markers', 
    name = 'Min Speed (m/s)', 
    yaxis = 'y2', 
    marker_color = 'orange'
))


SR_fig.update_layout(
    yaxis=dict(
        title='Rate', 
        range = [15, 45]
    ),
    yaxis2=dict(
        title='Speed',
        overlaying='y',
        side='right'
    )
)
selected_points = plotly_events(
    SR_fig,
    click_event=False,
    select_event=True,
    hover_event=False,
    override_height=500  # Adjust if you need a different figure height
)
if selected_points:
    # Extract the x-values of the selected points
    selected_x_vals = [pt["x"] for pt in selected_points]
    stream_start = selected_x_vals[0]
    stream_end = selected_x_vals[-1]
    stroke_on = np.where(stroke_data['onset_index'] == stream_start)[0][0]
    stroke_off = np.where(stroke_data['onset_index'] == stream_end)[0][0]
    
else: 
    selected_x_vals = list(range(0,len(stroke_data)))
    stream_start = 0
    stream_end = -1
    stroke_on = 0
    stroke_off = -1

SR_fig.update_layout(title = '<b>Select Data Range For Analysis</b>')

#Analyzing Cropped Data
stroke_data = stroke_data.iloc[stroke_on:stroke_off].reset_index(drop=True)



# Now you can group by Rate_Bin and do further analysis, for example:
DPS_group = stroke_data.groupby('Rate_Bin')['DPS'].mean()
dist_group = stroke_data.groupby('Rate_Bin')['DPS'].sum()
speed_group = stroke_data.groupby('Rate_Bin')['speed'].mean()
min_group = stroke_data.groupby('Rate_Bin')['min speed'].mean()
max_group = stroke_data.groupby('Rate_Bin')['max speed'].mean()
eWPS_group = stroke_data.groupby('Rate_Bin')['eWPS'].mean()

#training summary datatable for all data, regardless of cropping

train_summary = pd.DataFrame()
train_summary['DPS'] = round(DPS_group,2)
train_summary['Stroke Speed'] = round(speed_group,2)
train_summary['Min Stroke Speed'] = round(min_group,2)
train_summary['Max Stroke Speed'] = round(max_group,2)
train_summary['eWPS'] = round(eWPS_group,2) 
train_summary['Distance in Range'] = round(dist_group,2)
train_summary = train_summary.T
train_summary = train_summary.reset_index()

train_summary = train_summary.rename(columns={'index': 'Metric'})


#data = data.iloc[stream_start:stream_end].reset_index(drop=True)
up_frame = up_frame.iloc[stream_start:stream_end].reset_index(drop=True)

#st.line_chart(up_frame['speed'])
#st.line_chart(data['speed'].iloc[int(stream_start/factor):int((stream_end+200)/factor)].reset_index(drop=True))
#st.write(data['gnss.timestamp'].iloc[int(stream_end/factor)]- data['gnss.timestamp'].iloc[int(stream_start/factor)])
#st.write( data['gnss.timestampPC'].iloc[int(stream_start/factor)].split('-')[0:-1])
#st.write( data['gnss.timestampPC'].iloc[int(stream_end/factor)].split('-')[0:-1])


stroke_data['Distance Piece'] = round(stroke_data['Distance'] - stroke_data['Distance'].iloc[0])
max_dist = stroke_data['Distance Piece'].max()
interval_step = 250
intervals = round(max_dist/interval_step)


interval_df = pd.DataFrame()
interval_distances = []
for interval in range(0,intervals): 
    int_data = stroke_data[
    (100 * interval <= stroke_data['Distance Piece']) &
    (stroke_data['Distance Piece'] <= (interval_step * interval + interval_step))
    ]
    if not int_data.empty:
        interval_distances.append(interval_step * interval + interval_step)
        interval_df.loc[f'{interval_step * interval + interval_step}m','Speed (m/s)'] = round(int_data['speed'].mean(),2)
        interval_df.loc[f'{interval_step * interval + interval_step}m','Split (/500m)'] = speed_to_split(int_data['speed'].mean())
        interval_df.loc[f'{interval_step * interval + interval_step}m','Min Speed (m/s)'] = round(int_data['min speed'].mean(),2)
        interval_df.loc[f'{interval_step * interval + interval_step}m','Max Speed (m/s)'] = round(int_data['max speed'].mean(),2)
        interval_df.loc[f'{interval_step * interval + interval_step}m','Rate (SPM)'] = round(int_data['Rate'].mean(),1)
        interval_df.loc[f'{interval_step * interval + interval_step}m','eWPS'] = round(int_data['eWPS'].mean(),2)
        interval_df.loc[f'{interval_step * interval + interval_step}m','DPS (m)'] = round(int_data['DPS'].mean(),2)
    

interval_df = interval_df.reset_index()


accel_arrrays = []
accel_fig = go.Figure()
for val in range(1,len(stroke_data['onset_index'])): 

    if (stroke_data['onset_index'][val]- stroke_data['onset_index'][val-1])<=int(40*factor):
        

        accel_env = upsample_accel[stroke_data['onset_index'][val-1]:stroke_data['onset_index'][val]]
        accel_env = lowpass(accel_env,4,10*factor)
        accel_fig.add_trace(go.Scattergl(
            y = accel_env, 
            mode = 'lines', 
            marker_color = 'blue', 
            opacity = .05,
        ))

        accel_arrrays.append(pd.Series(accel_env).values)

if accel_arrrays:
    # Determine the maximum stride length
    max_len = max(len(arr) for arr in accel_arrrays)

    # Pad each stride to max_len_r
    padded_accel = []
    for arr in accel_arrrays:
        # Calculate how many NaNs we need
        pad_len = max_len - len(arr)
        # Pad at the end with NaN
        padded_arr = np.pad(arr, (0, pad_len), mode='constant', constant_values=np.nan)
        padded_accel.append(padded_arr)

    # Convert to an array of shape (num_strides, max_stride_len)
    padded_accel = np.array(padded_accel)  

    # Compute average ignoring NaNs
    avg_stroke = np.nanmean(padded_accel, axis=0)

    # Plot the average stride
    accel_fig.add_trace(go.Scattergl(
        y=avg_stroke,
        line=dict(color='red', width=5),  # thicker line
    ))



accel_fig.update_traces(showlegend=False) 
accel_fig.update_layout(xaxis_title = '<b>Sample Number</b>')
accel_fig.update_layout(yaxis_title = '<b>Acceleration</b> (m/s^2)')
accel_fig.update_layout(title = '<b>Average Stroke Acceleration Profile</b>')



stroke_number = 0
stroke_min_vel = []
stroke_max_vel = []
stroke_mean_vel = []
stroke_mean_accel = []



stroke_plot = go.Figure()

for val in range(1,len(stroke_data['onset_index'])): 

    if (stroke_data['onset_index'][val] - stroke_data['onset_index'][val-1])<=int(40*factor):
        stroke_number += 1
        

        

        #speed_env = speed_plot[stroke_data['onset_index'][val-1]:stroke_data['onset_index'][val]]
        speed_env = upsample_speed[stroke_data['onset_index'][val-1]:stroke_data['onset_index'][val]]


        #Upsampling
        factor_new = 3
        original_length_new = len(speed_env)
        
        if original_length_new > 2:
                
            # Old indices: 0, 1, 2, ..., original_length - 1
            old_indices_new = np.arange(original_length_new)
            
            # New indices span from 0 to original_length - 1, with factor * original_length points
            new_indices_new = np.linspace(0, original_length_new - 1, int(factor_new * original_length_new))
            
            # Perform linear interpolation
            speed_env = np.interp(new_indices_new, old_indices_new, speed_env)
            speed_env = lowpass(speed_env, 2,10)
            accel_env_plot  = np.diff(speed_env)/(1/(10*factor*factor_new))
            accel_env_plot = lowpass(accel_env_plot, 4*factor_new, 10*factor*factor_new)
            
            stroke_plot.add_trace(go.Scattergl(
                y = [stroke_number]*len(speed_env), 
                x = np.array(list(range(0,len(speed_env))))/(10*factor*factor_new),
                mode = 'markers', 
                marker=dict(
                    size=4,
                    color=accel_env_plot,  # Set color to velocity
                    colorscale='Viridis', 
                    cmin=-8,  # Fixed minimum value of your color scale
                    cmax=9.8, # Choose a color scale
                    coloraxis="coloraxis",  # Reference to a shared color axis
                    opacity=1
                )
            ))
# Update layout to include a shared coloraxis (colorbar) settings
stroke_plot.update_layout(
    title="Stacked Stroke Visualization",
    xaxis_title="Stroke time (s)",
    yaxis_title="Stroke Number",
    showlegend=False,  # This line removes the legend
    yaxis=dict(tickmode='array'),
    coloraxis=dict(
        cmin= -9.81, 
        cmax = 9.81,
        colorbar=dict(title="Acceleration (m/s2)"),
        colorscale='Viridis'
    )
)
#st.plotly_chart(stroke_plot)

_='''
DPS vs stroke rate profiling

- Rationale here is that DPS and SR combine to boat velocity. This in theory should identify the vertex relative to stroke rate
- Find theoretical max velocity for boats
- Beautiful thing here is that we can compare this to the world rowing data
'''

peak_vals = []
rate_list = []
AV_data = stroke_data[stroke_data['Rate']>15]
AV_data = AV_data[AV_data['DPS']<20]

for rate in list(range(17, AV_data['SR Round'].max()+1)):
    
    try: 
        peak = boxplot_upper_whisker(AV_data['DPS'][AV_data['SR Round'] == rate])
        peak_vals.append(peak)
        rate_list.append(rate)
    except:
        pass

slope, intercept, r, p, se = linregress(rate_list, peak_vals)
max_rate = -(intercept/slope)


_='''
AV_fig = make_subplots(specs=[[{"secondary_y": True}]])
AV_fig.add_trace(go.Scattergl(
              y = stroke_data['DPS'][stroke_data['Rate']>15],#abs(stroke_data['accel']),
              x = stroke_data['Rate'][stroke_data['Rate']>15],
              name = 'DPS', #stroke_data['speed'], 
              mode = 'markers'))


AV_fig.add_trace(go.Scattergl(
              y = (stroke_data['DPS'][stroke_data['Rate']>15]*stroke_data['Rate'][stroke_data['Rate']>15])/60,#abs(stroke_data['accel']),
              x = stroke_data['Rate'][stroke_data['Rate']>15],#stroke_data['speed'], 
              name = 'DPS, Rate', 
              mode = 'markers'), 
              secondary_y=True)


AV_fig.add_trace(go.Scattergl(
              y = peak_vals,#abs(stroke_data['accel']),
              x = rate_list,#stroke_data['speed'], 
              name = 'Bin Maximums', 
              mode = 'markers'))

AV_fig.add_trace(go.Scattergl(
              y = (slope*np.array(range(15,55)) + intercept),
              x = np.array(range(15,int(max_rate)+1)),#stroke_data['speed'], 
              name = 'Regression', 
              mode = 'lines'))
AV_fig.add_trace(go.Scattergl(
              y = (slope*np.array(range(15,55)) + intercept)*np.array(range(15,55))/60,
              x = np.array(range(15,55)),#stroke_data['speed'], 
              name = 'Parabola', 
              mode = 'lines'), 
              secondary_y=True)


AV_fig.add_trace(go.Scattergl(
              y = (np.array(rate_list)*np.array(peak_vals))/60,#abs(stroke_data['accel']),
              x = rate_list,#stroke_data['speed'], 
              name = 'Bin Maximums Rates', 
              mode = 'markers'), 
              secondary_y=True)


parab = np.array(slope*np.array(range(15,int(max_rate)+1)) + intercept)*np.array(range(15,int(max_rate)+1))

AV_fig.add_shape(
    type="line",
    xref="x",       # Use the x-axis reference
    x0=np.where(parab == np.max(parab))[0][0]+15, x1=np.where(parab == np.max(parab))[0][0]+15,   # Start and end at x=41
    yref="paper",   # Use the entire paper height
    y0=0, y1=1,     # Span from bottom to top
    line=dict(
        dash="dot",  # Dotted line
        color="black"
    )
)



AV_fig.update_layout(
    title="Rate and Distance Per Stroke Profiling",
    xaxis_title="Stroke Rate (SPM)",
)

AV_fig.update_yaxes(title_text="DPS (m)", secondary_y=False)
AV_fig.update_yaxes(title_text="Effective Power", secondary_y=True)
AV_fig.update_yaxes(range=[5, 20], secondary_y=False)  # Primary y-axis range
AV_fig.update_yaxes(range=[0, 10], secondary_y=True) 
AV_fig.update_xaxes(range=[15, 50]) 




av_plot, av_metrics = st.columns([4,2])
with av_plot:
    st.plotly_chart(AV_fig)
with av_metrics: 
    st.metric('Slope', round(slope, 3))
    st.metric('Theo Max Speed', round(np.max(parab)/60, 2))
    st.metric('Sweet Spot',round(np.where(parab == np.max(parab))[0][0]+15, 2))
'''

_='''
calculating averages for display
'''

average_speed = np.mean(stroke_data['speed'])
average_split = speed_to_split(average_speed)
max_rate = np.max(stroke_data['Rate'])
min_rate = np.min(stroke_data['Rate'])
average_rate = np.mean(stroke_data['Rate'])
average_eWPS = np.mean(stroke_data['eWPS'])
average_DPS = np.mean(stroke_data['DPS'])
section_dist = stroke_data['Distance'].iloc[-1] - stroke_data['Distance'].iloc[0]
dps_distance = np.sum(stroke_data['DPS'])



metric1, metric2, metric3, metric4 = st.columns([1,1,1,1])

with metric1: 
    st.header('Speeds')
    st.metric('Average Speed (m/s), (Prog)', f'{round(average_speed,2)}, {round(average_speed/float(prog_vel) *100,1)}%')
    st.metric('Average Split (/500m)', average_split)
with metric2:
    st.header('Rate')
    st.metric('Average Rate (SPM)', round(average_rate,2))
    st.metric('Rate Range (SPM)', f'{round(max_rate,1)}, {round(min_rate,1)}')
    #st.metric('Min Rate (SPM)', round(min_rate,2))
with metric3:
    st.header('Energy')
    st.metric('Average eWPS', round(average_eWPS,2))
    st.metric('Average ePow', round(average_eWPS*average_rate,2))
with metric4:
    st.header('Distances')
    st.metric('Average DPS (m)', round(average_DPS,2))
    st.metric('Section Distance (m)(speed)', round(np.mean(up_frame['speed'])*len(up_frame)/10/factor,2))

st.metric('Section Time', sec_to_split(dps_distance/np.mean(up_frame['speed'])))

    


speed = up_frame['speed']
lat = up_frame['lat']
long = up_frame['long']
map = go.Figure(go.Scattermapbox(
		lon=long,
		lat=lat,
	    mode='markers',
    marker=go.scattermapbox.Marker(
            size=4,
            color=(speed/float(prog_vel))*100,            # Use speed array for color
            colorscale="Jet",   # Any valid Plotly colorscale (e.g., "Jet", "Turbo", "Viridis")
            showscale=True, 
            cmin=50,   # Fixed minimum color value
            cmax=100,        # Show colorbar
            colorbar=dict(
                title="Speed",      # Title for the colorbar
                x=0.95,            # Adjust x-position of colorbar if needed
                y=0.5,
                #titlefont=dict(color='white'),  # Change title font color to white
                tickfont=dict(color='white'),   # Change tick font color to white
                len=0.8
            )),

	    textposition="top center",
	    hoverinfo="text"))
map.update_layout(
    mapbox={
            "style": "white-bg",  # We'll override the background with a custom layer
            "center": {"lat": np.mean(lat), "lon": np.mean(long)},
            "zoom": 13,
            # Add a custom tile layer for satellite imagery from ArcGIS
            "layers": [
                {
                    "below": "traces",  
                    "sourcetype": "raster",
                    "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                    "source": [
                        "https://services.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
    },
	    title="Row Session Location Trace",
        
	    margin={"r":0, "t":0, "l":0, "b":0}  # Remove margins
	)

map_box, accel_box = st.columns([2,1])
with map_box:
    #
    st.plotly_chart(stroke_plot)


with accel_box:
    st.plotly_chart(accel_fig)

st.plotly_chart(map)    
#map.show()




_='''
Generation of a PDF for Training session Reporting

'''

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt
from io import BytesIO



def training_rep():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    margin_left = 50
    margin_top = 50
    top_position = page_height - margin_top

    image_width = 50
    image_height = 50
    c.drawImage(
        "https://images.squarespace-cdn.com/content/v1/5f63780dbc8d16716cca706a/1604523297465-6BAIW9AOVGRI7PARBCH3/rowing-canada-new-logo.jpg",
        x=margin_left,
        y=top_position - image_height,
        width=image_width,
        height=image_height,
        preserveAspectRatio=True
    )

    title_text = "Training Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left + image_width + 10, top_position - image_height/2, title_text)

    num1 = round(session_distance / 1000, 2)
    num2 = sec_to_split(session_time)
    numbers_y = top_position - image_height - 20

    c.setFont("Helvetica", 12)
    c.drawString(margin_left, numbers_y, f"Total Session Distance (km): {num1}")
    c.drawString(margin_left , numbers_y-20, f"Total Session Time (mm:ss.ms): {num2}")

    columns = train_summary.columns.tolist()
    rows = train_summary.values.tolist()
    data_for_reportlab = [columns] + rows

    col_count = len(columns)
    col_widths = [.60 * inch] * (col_count-1)
    col_widths.insert(0,1.5*inch)

    table = Table(data_for_reportlab, colWidths=col_widths)
    style = TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEABOVE", (0, 1), (-1, 1), 1, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ])
    table.setStyle(style)

    _, table_height = table.wrapOn(c, page_width, page_height)
    table_x = margin_left
    table_y = top_position - image_height - 60 - table_height
    table.drawOn(c, table_x, table_y)

    # ---- Generate an in-memory Matplotlib plot ----
    #fig, ax = plt.subplots()
    #ax.scatter(stroke_data['Rate'][stroke_data['Rate']>15], stroke_data['DPS'][stroke_data['Rate']>15])
    #ax.scatter(stroke_data['Rate'][stroke_data['Rate']>15], stroke_data['DPS'][stroke_data['Rate']>15]*stroke_data['Rate'][stroke_data['Rate']>15])   # Simple example data
    #ax.set_title("Distance Per Stroke Over Rates")

    # Save plot to an in-memory BytesIO buffer
    #plot_buffer = BytesIO()
    #fig.savefig(plot_buffer, format='PNG', dpi=72)  # Adjust DPI if needed
    #plot_buffer.seek(0)
    #plt.close(fig)  # Close figure to free memory

    # Convert the BytesIO data into a ReportLab-compatible image
    #plot_img = ImageReader(plot_buffer)

    # 6. Place the plot below the table
    # Figure out how far down to place the plot
    #plot_width = 500
    #plot_height = 200
    #bottom_of_table = table_y
    #y_for_plot = bottom_of_table - plot_height - 20
    '''
    c.drawImage(
        plot_img,
        margin_left,
        y_for_plot,
        width=plot_width,
        height=plot_height,
        preserveAspectRatio=True
    )
    '''

    c.save()
    buffer.seek(0)
    return buffer






_='''
Generation of a PDF for Race Reporting

'''

def race_rep():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    margin_left = 50
    margin_top = 50
    top_position = page_height - margin_top

    image_width = 50
    image_height = 50
    c.drawImage(
        "https://images.squarespace-cdn.com/content/v1/5f63780dbc8d16716cca706a/1604523297465-6BAIW9AOVGRI7PARBCH3/rowing-canada-new-logo.jpg",
        x=margin_left,
        y=top_position - image_height,
        width=image_width,
        height=image_height,
        preserveAspectRatio=True
    )

    title_text = "Race Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left + image_width + 10, top_position - image_height/2, title_text)

    num1 = round(np.max(interval_distances))
    num2 = speed_to_split_dist(interval_df['Speed (m/s)'].mean(), np.max(interval_distances))
    numbers_y = top_position - image_height - 20

    c.setFont("Helvetica", 12)
    c.drawString(margin_left, numbers_y, f"Race Distance: {num1}")
    c.drawString(margin_left , numbers_y-20, f"Race Time (mm:ss.ms): {num2}")

    columns_race = interval_df.columns.tolist()
    rows_race = interval_df.values.tolist()
    data_for_report = [columns_race] + rows_race

    col_count = len(columns_race)
    col_widths = [0.9 * inch] * (col_count-1)
    col_widths.insert(0, .5 * inch)

    table = Table(data_for_report, colWidths=col_widths)
    style = TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEABOVE", (0, 1), (-1, 1), 1, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ])
    table.setStyle(style)

    _, table_height = table.wrapOn(c, page_width, page_height)
    table_x = margin_left
    table_y = top_position - image_height - 60 - table_height
    table.drawOn(c, table_x, table_y)

    c.save()
    buffer.seek(0)
    return buffer




file_name = st.sidebar.text_input('Input file name', 'Enter Report Details')

if st.sidebar.button("Generate Training Report"):
    pdf_data = training_rep()
    st.sidebar.download_button(
        label="Download Training Report",
        data=pdf_data,
        file_name=f"{file_name}_training_report.pdf",
        mime="application/pdf"
    )

if st.sidebar.button("Generate Race Report"):
    pdf_data = race_rep()
    st.sidebar.download_button(
        label="Download Race Report",
        data=pdf_data,
        file_name=f"{file_name}_race_report.pdf",
        mime="application/pdf"
    )
