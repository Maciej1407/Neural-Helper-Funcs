import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scst
from scipy.signal import correlate, correlation_lags, curve_fit

def load_data(data_dir="Data/"):
    """Loads and preprocesses data from CSV and NPY files."""
    frame_t_df = pd.read_csv(os.path.join(data_dir, "frameTimes.csv"), sep=',', header=None).T
    frame_t = frame_t_df.to_numpy()[:, 0]

    stimIDs_df = pd.read_csv(os.path.join(data_dir, "stimIDs.csv"), sep=',', header=None).T
    stimIDs = stimIDs_df.to_numpy()[:, 0]

    StimOn_df = pd.read_csv(os.path.join(data_dir, "stimOnsetTimes.csv"), sep=',', header=None).T
    StimOn = StimOn_df.to_numpy()[:, 0]

    StimOff_df = pd.read_csv(os.path.join(data_dir, "stimOffsetTimes.csv"), sep=',', header=None).T
    StimOff = StimOff_df.to_numpy()[:, 0]

    FluoData = np.load(os.path.join(data_dir, "Suite2p/plane0/F.npy"))
    fDataGood = np.load(os.path.join(data_dir, "Suite2p/plane0/iscell.npy"))

    validity_mask = fDataGood[:, 0] == 1.0
    filtered_data = FluoData[validity_mask, :]
    filtered_pd = pd.DataFrame(filtered_data.T)
    filtered_data_z = scst.zscore(filtered_data, axis=1)

    return frame_t, stimIDs, StimOn, StimOff, FluoData, filtered_data, filtered_pd, filtered_data_z

def plot_neuronal_activity(time, neuronal_activity, StimOn, StimOff, neurons_to_plot=3, time_window=1000, stim_num=10):
    """Plots neuronal activity with stimulus periods."""
    indicies = [i for i in range(0, neuronal_activity.shape[0])]
    rand_indicies = np.random.choice(indicies, neurons_to_plot, replace=False)
    neurons_plot = neuronal_activity[rand_indicies]

    fig = go.Figure()
    for i in range(neurons_to_plot):
        fig.add_trace(go.Scatter(x=time[:time_window], y=neurons_plot[i][:time_window], mode='lines', opacity=0.5, name=f'Neuron {rand_indicies[i]}'))

    for onset, offset in zip(StimOn[:stim_num], StimOff[:stim_num]):
        fig.add_trace(go.Scatter(x=[onset, offset, offset, onset], y=[np.min(neuronal_activity[:neurons_to_plot, :time_window]), np.min(neuronal_activity[:neurons_to_plot, :time_window]), np.max(neuronal_activity[:neurons_to_plot, :time_window]), np.max(neuronal_activity[:neurons_to_plot, :time_window])], mode='lines', fill="toself", fillcolor="red", opacity=0.2, line=dict(width=0), name="Stimulus Period", showlegend=False if (onset != StimOn[0]) else True))
        fig.add_trace(go.Scatter(x=[onset, onset], y=[np.min(neuronal_activity[:neurons_to_plot, :time_window]), np.max(neuronal_activity[:neurons_to_plot, :time_window])], mode='lines', line=dict(color='red', dash='dash', width=1), opacity=0.8, showlegend=False))
        fig.add_trace(go.Scatter(x=[offset, offset], y=[np.min(neuronal_activity[:neurons_to_plot, :time_window]), np.max(neuronal_activity[:neurons_to_plot, :time_window])], mode='lines', line=dict(color='blue', dash='dash', width=1), opacity=0.8, showlegend=False))

    fig.update_layout(title='Neuronal Firing Rates with Stimulus Periods', xaxis_title='Time (s)', yaxis_title='Firing Rate (normalized)')
    fig.show()

def plot_individual_avg(filtered_data_z, avg_filt, frame_t, StimOn, StimOff, t=5000, num_neurons_to_plot=6, num_stims=40, type="sum", title='Neuronal Firing Rates with Stimulus Periods', xaxis_title='Time (s)', yaxis_title='Firing Rate (normalized)'):
    """Plots individual neuron firing rates and their average."""
    num_neurons = filtered_data_z.shape[0]
    rand_indices = np.random.choice(range(num_neurons), num_neurons_to_plot, replace=False)
    neurons_to_plot = filtered_data_z[rand_indices]

    if type == "sum":
        y_min = np.min(avg_filt[:t])
        y_max = np.max(avg_filt[:t])
    elif type == "avg":
        y_min = np.min(neurons_to_plot[:, :t])
        y_max = np.max(neurons_to_plot[:, :t])

    fig = go.Figure()
    for i in range(num_neurons_to_plot):
        fig.add_trace(go.Scatter(x=frame_t[:t], y=neurons_to_plot[i, :t], mode='lines', opacity=0.5, name=f'Neuron {rand_indices[i]}'))

    fig.add_trace(go.Scatter(x=frame_t[:t], y=avg_filt[:t], mode='lines', name="Avg", line=dict(color="black")))

    for onset, offset in zip(StimOn[:num_stims], StimOff[:num_stims]):
        fig.add_trace(go.Scatter(x=[onset, offset, offset, onset], y=[y_min, y_min, y_max, y_max], mode='lines', fill="toself", fillcolor="red", opacity=0.2, line=dict(width=0), name="Stimulus Period", showlegend=False if (onset != StimOn[0]) else True))
        fig.add_trace(go.Scatter(x=[onset, onset], y=[y_min, y_max], mode='lines', line=dict(color='red', dash='dash', width=1), opacity=0.8, showlegend=False))
        fig.add_trace(go.Scatter(x=[offset, offset], y=[y_min, y_max], mode='lines', line=dict(color='blue', dash='dash', width=1), opacity=0.8, showlegend=False))

    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()

def normalize_pcnt_act(data):
    """Normalizes neuron firing rate data to percentage of maximum firing rate."""
    num_neurons = data.shape[0]
    normalized_arr = np.zeros_like(data, dtype=float)
    for i in range(num_neurons):
        max_val = np.max(data[i, :])
        if max_val != 0:
            normalized_arr[i, :] = (data[i, :] / max_val) * 100
        else:
            normalized_arr[i, :] = 0
    return normalized_arr

def filter_stimuli(StimOn, StimOff, frame_t_subsection, leeway_time):
    """Filters StimOn and StimOff arrays to keep only values within a specified time range."""
    start_time = frame_t_subsection[0] - leeway_time
    end_time = frame_t_subsection[-1] + leeway_time
    filtered_StimOn = StimOn[(StimOn >= start_time) & (StimOn <= end_time)]
    filtered_StimOff = StimOff[(StimOff >= start_time) & (StimOff <= end_time)]
    return filtered_StimOn, filtered_StimOff
def plot_neuronal_activity_avg_sum(filtered_data, frame_t, StimOn, StimOff, mode='avg', time_range="full"):
    """
    Plots neuronal activity with stimulus markers.

    Args:
        filtered_data (np.ndarray): Neuron firing rate data (neurons x time).
        frame_t (np.ndarray): Time axis.
        StimOn (np.ndarray): Array of stimulus onset times.
        StimOff (np.ndarray): Array of stimulus offset times.
        mode (str, optional): 'avg' for average activity, 'sum' for summed activity. Defaults to 'avg'.
        time_range (str, tuple, int, optional): Time range to plot. 
            'full' for the entire time range, 
            a tuple (start_time, end_time) for a specific time interval,
            or an integer N for the last N samples. Defaults to "full".

    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """

    # 1. Time Range Selection:
    if time_range == "full":
        start_index = 0
        end_index = len(frame_t)
    elif isinstance(time_range, tuple):
        start_time, end_time = time_range
        start_index = np.argmin(np.abs(frame_t - start_time))
        end_index = np.argmin(np.abs(frame_t - end_time)) + 1
    elif isinstance(time_range, int):  # Last N samples
        start_index = len(frame_t) - time_range
        end_index = len(frame_t)
    else:
        raise ValueError("Invalid time_range. Must be 'full', a tuple, or an integer.")

    time_segment = frame_t[start_index:end_index]
    data_segment = filtered_data[:, start_index:end_index]

    # 2. Calculate Average/Summed Activity:
    if mode == 'avg':
        activity = np.mean(data_segment, axis=0)
    elif mode == 'sum':
        activity = np.sum(data_segment, axis=0)
    else:
        raise ValueError("Invalid mode. Must be 'avg' or 'sum'.")

    # 4. Filter Stimuli within Time Range:
    filtered_StimOn = [onset for onset in StimOn if time_segment[0] <= onset <= time_segment[-1]]
    filtered_StimOff = [offset for offset in StimOff if time_segment[0] <= offset <= time_segment[-1]]

    # 5. Plotting:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_segment, y=activity, mode='lines', name='Average Activity'))

    for onset, offset in zip(filtered_StimOn, filtered_StimOff):
        min_y = np.min(activity) if len(activity) > 0 else 0
        max_y = np.max(activity) if len(activity) > 0 else 1

        fig.add_trace(go.Scatter(
            x=[onset, offset, offset, onset],
            y=[min_y, min_y, max_y, max_y],
            mode='lines',
            fill="toself",
            fillcolor="red",
            opacity=0.2,
            line=dict(width=0),
            name="Stimulus Period",
            showlegend=False if (onset != StimOn[0]) else True
        ))
        fig.add_trace(go.Scatter(x=[onset, onset], y=[min_y, max_y],
                         mode='lines', line=dict(color='red', dash='dash', width=1), opacity=0.8, showlegend=False))
        fig.add_trace(go.Scatter(x=[offset, offset], y=[min_y, max_y],
                         mode='lines', line=dict(color='blue', dash='dash', width=1), opacity=0.8, showlegend=False))

    title = "Average" if mode == 'avg' else "Summed"
    
    fig.update_layout(
        title=f'{title} Neuronal Firing Rate with Stimulus Periods',
        xaxis_title='Time (s)',
        yaxis_title='Firing Rate '
    )

    return fig



def segment_data(filt_data, frame_t, StimOn, StimOff, pre_stim_samples=10, post_stim_samples=10):
    """
    Segments neuronal data around stimulus periods.

    Args:
        filt_data (np.ndarray): Neuronal firing rate data (neurons x time).
        frame_t (np.ndarray): Time array.
        StimOn (np.ndarray): Stimulus onset times.
        StimOff (np.ndarray): Stimulus offset times.
        pre_stim_samples (int, optional): Number of samples to include before stimulus onset. Defaults to 10.
        post_stim_samples (int, optional): Number of samples to include after stimulus offset. Defaults to 10.

    Returns:
        dict: A dictionary where keys are stimulus indices and values are 
              dictionaries containing the segmented data and corresponding time array.
              Returns an empty dictionary if no valid segments are found.
    """
    segmented_data = {}

    for i, (onset, offset) in enumerate(zip(StimOn, StimOff)):
        onset_index = np.argmin(np.abs(frame_t - onset))
        offset_index = np.argmin(np.abs(frame_t - offset))

        start_index = max(0, onset_index - pre_stim_samples)
        end_index = min(filt_data.shape[1] - 1, offset_index + post_stim_samples)

        segment = filt_data[:, start_index:end_index]
        segment_time = frame_t[start_index:end_index]

        if segment.size > 0:
            segmented_data[i] = {'data': segment, 'time': segment_time}
        else:
            print(f"Warning: No valid segment found for stimulus {i}. Skipping.")

    return segmented_data

def segment_data_2(filt_data, frame_t, StimOn, StimOff, pre_stim_samples=5, time_to_next_stim=10):
    """
    Segments neuronal data around stimulus periods, leading up to a 
    specified time before the next stimulus.

    Args:
        filt_data (np.ndarray): Neuronal firing rate data (neurons x time).
        frame_t (np.ndarray): Time array.
        StimOn (np.ndarray): Stimulus onset times.
        StimOff (np.ndarray): Stimulus offset times.
        pre_stim_samples (int, optional): Number of samples to include before stimulus onset. Defaults to 5.
        time_to_next_stim (int, optional): Time (in samples) before the *next* stimulus 
                                           to end the current segment. Defaults to 10.

    Returns:
        dict: A dictionary where keys are stimulus indices and values are 
              dictionaries containing the segmented data, corresponding time array,
              and the actual StimOn and StimOff times for that segment.
              Returns an empty dictionary if no valid segments are found.
    """
    segmented_data = {}

    for i, (onset, offset) in enumerate(zip(StimOn, StimOff)):
        onset_index = np.argmin(np.abs(frame_t - onset))
        offset_index = np.argmin(np.abs(frame_t - offset))

        start_index = max(0, onset_index - pre_stim_samples)

        if i < len(StimOn) - 1:
            next_stim_time = StimOn[i + 1]
            end_time = next_stim_time - frame_t[0]
            end_index = np.argmin(np.abs(frame_t - (offset + (end_time - offset))))

        else:
            end_index = filt_data.shape[1] - 1

        segment = filt_data[:, start_index:end_index]
        segment_time = frame_t[start_index:end_index]

        if segment.size > 0:
            segmented_data[i] = {
                'data': segment,
                'time': segment_time,
                'stim_on': onset,
                'stim_off': offset,
            }
        else:
            print(f"Warning: No valid segment found for stimulus {i}. Skipping.")

    return segmented_data

import plotly.graph_objects as go
import numpy as np

def plot_segmented_neuron_activity(segmented_data, max_plots=2):
    """
    Plots individual neuron activity for a specified number of segments, 
    including stimulus onset and offset markers.

    Args:
        segmented_data (dict): A dictionary where keys are segment indices and values are 
                               dictionaries containing 'data', 'time', 'stim_on', and 'stim_off'.
        max_plots (int, optional): Maximum number of segments to plot. Defaults to 2.
    """
    count = 0
    for i, seg_info in segmented_data.items():
        if count == max_plots:
            break
        data = seg_info['data']
        time = seg_info['time']
        stim_on = seg_info['stim_on']
        stim_off = seg_info['stim_off']

        fig = go.Figure()
        for neuron in range(data.shape[0]):
            fig.add_trace(go.Scatter(x=time, y=data[neuron, :], mode='lines', name=f'Neuron {neuron + 1}'))

        min_y = np.min(data)
        max_y = np.max(data)

        fig.add_trace(go.Scatter(x=[stim_on, stim_on], y=[min_y, max_y], mode='lines', line=dict(color='red', dash='dash', width=1), name="Stim On", showlegend=False))
        fig.add_trace(go.Scatter(x=[stim_off, stim_off], y=[min_y, max_y], mode='lines', line=dict(color='blue', dash='dash', width=1), name="Stim Off", showlegend=False))

        fig.update_layout(title=f"Segment {i} Activity - Individual Neurons", xaxis_title="Time (s)", yaxis_title="Firing Rate")
        fig.show()

        count += 1


def segment_data_multiple_stimuli(filt_data, frame_t, StimOn, StimOff, pre_stim_samples=5, post_stim_samples=5, num_stimuli_per_segment=1):
    """
    Segments neuronal data around multiple stimulus periods.

    Args:
        filt_data (np.ndarray): Neuronal firing rate data (neurons x time).
        frame_t (np.ndarray): Time array.
        StimOn (np.ndarray): Stimulus onset times.
        StimOff (np.ndarray): Stimulus offset times.
        pre_stim_samples (int, optional): Samples before the first StimOn. Defaults to 5.
        post_stim_samples (int, optional): Samples after the last StimOff. Defaults to 5.
        num_stimuli_per_segment (int, optional): Number of StimOn/Off pairs per segment. Defaults to 1.

    Returns:
        dict: Segmented data (keys are segment indices, values are dictionaries
              containing 'data', 'time', 'stim_on', 'stim_off').
              Returns an empty dictionary if no valid segments are found.
    """
    segmented_data = {}
    segment_index = 0

    for i in range(0, len(StimOn), num_stimuli_per_segment):
        segment_stim_on = StimOn[i:min(i + num_stimuli_per_segment, len(StimOn))]
        segment_stim_off = StimOff[i:min(i + num_stimuli_per_segment, len(StimOff))]

        first_onset_index = np.argmin(np.abs(frame_t - segment_stim_on[0]))
        last_offset_index = np.argmin(np.abs(frame_t - segment_stim_off[-1]))

        start_index = max(0, first_onset_index - pre_stim_samples)
        end_index = min(filt_data.shape[1] - 1, last_offset_index + post_stim_samples)

        segment_data = filt_data[:, start_index:end_index]
        segment_time = frame_t[start_index:end_index]

        if segment_data.size > 0:
            segmented_data[segment_index] = {
                'data': segment_data,
                'time': segment_time,
                'stim_on': segment_stim_on,
                'stim_off': segment_stim_off,
            }
            segment_index += 1
        else:
            print(f"Warning: No valid segment found starting at stimulus {i}. Skipping.")

    return segmented_data

def sum_segmented_data(segmented_data):
    """
    Sums the neuronal activity across neurons for each segment in the given data.

    Args:
        segmented_data (dict): A dictionary where keys are segment indices and values are 
                               dictionaries containing 'data', 'time', 'stim_on', and 'stim_off'.

    Returns:
        dict: A new dictionary with the same keys as segmented_data, but with 'data'
              containing the summed neuronal activity for each segment.
    """
    summed_segments = {}
    for i, segment in segmented_data.items():
        summed_segments[i] = {
            'data': np.sum(segment['data'], axis=0),
            'time': segment['time'],
            'stim_on': segment['stim_on'],
            'stim_off': segment['stim_off'],
        }
    return summed_segments

def exponential_decay(t, a, b, c):
    """Exponential decay function."""
    return a * np.exp(-b * t) + c


def calculate_correlation_matrices(half1, half2, pop_avg1, pop_avg2, dt=0.032, max_lag_seconds=0.5):
    """
    Calculates correlation matrices with and without excluding self-contribution,
    and correlation function matrices, along with their respective lag times.

    Args:
        half1 (np.ndarray): First half of the neural data.
        half2 (np.ndarray): Second half of the neural data.
        pop_avg1 (np.ndarray): Population average for the first half.
        pop_avg2 (np.ndarray): Population average for the second half.
        dt (float, optional): Sampling interval in seconds. Defaults to 0.032.
        max_lag_seconds (float, optional): Maximum lag in seconds for correlation calculations. Defaults to 0.5.

    Returns:
        tuple: Correlation matrices (with and without exclusion), correlation function matrices,
               and their respective lag times.
    """
    L = int(max_lag_seconds / dt)
    lag_range = np.arange(-L, L + 1)

    corr_matrix1, lag_t1 = compute_corr_matrix(half1, pop_avg1, lag_range, dt, exclude_self=True)
    corr_matrix2, lag_t2 = compute_corr_matrix(half2, pop_avg2, lag_range, dt, exclude_self=True)

    corr_matrix1_wc, lag_t1_wc = compute_corr_matrix(half1, pop_avg1, lag_range, dt, exclude_self=False)
    corr_matrix2_wc, lag_t2_wc = compute_corr_matrix(half2, pop_avg2, lag_range, dt, exclude_self=False)

    corr_func_mat1, lags_func_mat1 = get_correlation_arrays(pop_avg1, half1, sampling_rate=dt, max_lag_seconds=max_lag_seconds)
    corr_func_mat2, lags_func_mat2 = get_correlation_arrays(pop_avg2, half2, sampling_rate=dt, max_lag_seconds=max_lag_seconds)

    return corr_matrix1, corr_matrix2, corr_matrix1_wc, corr_matrix2_wc, corr_func_mat1, corr_func_mat2, lag_t1, lag_t2, lag_t1_wc, lag_t2_wc, lags_func_mat1, lags_func_mat2

def calculate_com(corr_matrix, lag_times):
    """
    Calculates the center of mass (COM) for each neuron's correlation.

    Args:
        corr_matrix (np.ndarray): Correlation matrix (neurons x lags).
        lag_times (np.ndarray): Array of lag times.

    Returns:
        np.ndarray: Array of COM values for each neuron.
    """
    return np.sum(corr_matrix * lag_times, axis=1) / np.sum(corr_matrix, axis=1)

def create_correlation_dataframe(corr_matrix1, corr_matrix2, exclude_self=True):
    """
    Creates a Pandas DataFrame from flattened correlation matrices.

    Args:
        corr_matrix1 (np.ndarray): Correlation matrix for the first half.
        corr_matrix2 (np.ndarray): Correlation matrix for the second half.
        exclude_self (bool, optional): Whether self-contribution was excluded. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing flattened correlation values.
    """
    flat_cor1 = np.ndarray.flatten(corr_matrix1)
    flat_cor2 = np.ndarray.flatten(corr_matrix2)
    title_suffix = " (with excluded contribution of correlated neuron)" if exclude_self else " (with included contribution of correlated neuron)"
    return pd.DataFrame({
        "First Half": flat_cor1,
        "Second Half": flat_cor2
    }), title_suffix


def compute_corr_matrix(neural_data, pop_avg, lag_range, dt, exclude_self=False):
    """
    Compute the cross-correlation matrix for a given neural dataset and population average.
    """
    n_time_local = neural_data.shape[0]
    n_neurons = neural_data.shape[1]
    corr_matrix = np.zeros((n_neurons, len(lag_range)))
    
    for i, lag in enumerate(lag_range):
        if lag < 0:
            shifted_data = neural_data[-lag:, :]
            shifted_pop = pop_avg[:n_time_local + lag]
        elif lag > 0:
            shifted_data = neural_data[:-lag, :]
            shifted_pop = pop_avg[lag:]
        else:
            shifted_data = neural_data
            shifted_pop = pop_avg
        
        for n in range(n_neurons):
            if exclude_self:
                new_pop = shifted_pop - shifted_data[:, n] / n_neurons
                corr_matrix[n, i] = np.corrcoef(shifted_data[:, n], new_pop)[0, 1]
            else:
                corr_matrix[n, i] = np.corrcoef(shifted_data[:, n], shifted_pop)[0, 1]
    
    lag_times = lag_range * dt
    return corr_matrix, lag_times



def get_correlation_arrays(pop_avg, neural_data, sampling_rate=0.032, max_lag_seconds=0.5):
    """
    Computes correlation arrays for each neuron and the population average.
    """
    max_lag_samples = int(max_lag_seconds / sampling_rate)
    n_neurons = neural_data.shape[1]
    correlation_neuron = []
    lags_neuron = []
    
    for i in range(n_neurons):
        neuron_signal = neural_data[:, i]
        corr = correlate(pop_avg, neuron_signal, mode='full', method='auto')
        lags = correlation_lags(len(pop_avg), len(neuron_signal), mode='full')
        
        n_overlap = len(pop_avg) - np.abs(lags)
        corr_normalized = corr / n_overlap
        
        valid = np.abs(lags) <= max_lag_samples
        correlation_neuron.append(corr_normalized[valid])
        lags_neuron.append(lags[valid] * sampling_rate)
    
    return np.array(correlation_neuron), np.array(lags_neuron)



def plot_correlation_distributions(corr_df, title_suffix):
    """
    Plots box and violin plots for correlation distributions.

    Args:
        corr_df (pd.DataFrame): DataFrame containing correlation values.
        title_suffix (str): Suffix to append to the plot titles.
    """
    fig_box = px.box(corr_df, points="all", title=f"Correlation Coefficients two data splits{title_suffix}", labels={"variable": "Half", "value": "Correlation Coefficient"})
    fig_vio = px.violin(corr_df, box=True, points="all", title=f"Correlation Coefficients two data splits{title_suffix}", labels={"variable": "Half", "value": "Correlation Coefficient"})
    fig_box.show()
    fig_vio.show()

import pandas as pd
import plotly.express as px
import numpy as np

def plot_com_comparison(com1, com2, n_neurons):
    """
    Plots a scatter plot comparing COM values between the two halves.

    Args:
        com1 (np.ndarray): COM values for the first half.
        com2 (np.ndarray): COM values for the second half.
        n_neurons (int): Number of neurons.
    """
    df = pd.DataFrame({
        "COM_half1": com1,
        "COM_half2": com2,
        "Neuron": np.arange(n_neurons)
    })

    fig_scatter = px.scatter(
        df,
        x="COM_half1",
        y="COM_half2",
        hover_data=["Neuron"],
        labels={'COM_half1': 'COM Half 1 (s)', 'COM_half2': 'COM Half 2 (s)'},
        title='Center-of-Mass Comparison: Half 1 vs. Half 2',
        color="Neuron"
    )

    min_val = min(com1.min(), com2.min())
    max_val = max(com1.max(), com2.max())
    fig_scatter.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(dash="dash", color="gray")
    )

    fig_scatter.update_layout(width=600, height=600, xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(constrain="domain"))
    fig_scatter.show()