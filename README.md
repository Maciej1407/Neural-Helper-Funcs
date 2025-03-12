# Neural Analysis Helpers

This repository contains a Python module (`neural_analysis_helpers.py`) with a collection of helper functions for analyzing neural data. These functions facilitate tasks such as data loading, visualization, segmentation, and correlation analysis.

## Features

-   **Data Loading:**
    -   `load_data`: Loads and preprocesses neural data from CSV and NPY files.
-   **Visualization:**
    -   `plot_neuronal_activity`: Plots neuronal firing rates with stimulus periods.
    -   `plot_individual_avg`: Plots individual neuron firing rates and their average.
    -   `plot_neuronal_activity_avg_sum`: Plots average or summed neuronal activity with stimulus markers.
    -   `plot_segmented_neuron_activity`: Plots individual neuron activity for segmented data.
    -   `plot_correlation_distributions`: Plots box and violin plots for correlation distributions.
    -   `plot_com_comparison`: Plots a scatter plot comparing COM values between two data halves.
-   **Data Processing:**
    -   `normalize_pcnt_act`: Normalizes neuron firing rate data to a percentage of maximum firing rate.
    -   `filter_stimuli`: Filters stimulus onset and offset times within a specified range.
    -   `segment_data`: Segments neuronal data around stimulus periods.
    -   `segment_data_2`: Segments neuronal data up to a specified time before the next stimulus.
    -   `segment_data_multiple_stimuli`: Segments neuronal data around multiple stimulus periods.
    -   `sum_segmented_data`: Sums neuronal activity across neurons for each segment.
    -   `calculate_correlation_matrices`: Calculates correlation matrices and correlation function matrices.
    -   `calculate_com`: Calculates the center of mass (COM) for each neuron's correlation.
    -   `create_correlation_dataframe`: Creates a Pandas DataFrame from flattened correlation matrices.
    -   `compute_corr_matrix`: Computes the cross-correlation matrix.
    -   `get_correlation_arrays`: Computes correlation arrays for each neuron and the population average.
-   **Utility:**
    -   `exponential_decay`: Defines an exponential decay function.

## Getting Started

### Prerequisites

-   Python 3.x
-   Libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scipy`, `plotly`

Install the required libraries:

```bash
pip install numpy pandas seaborn matplotlib scipy plotly
```
### Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    ```

2.  Place `neural_analysis_helpers.py` in your project directory, or within a directory that is within your python path.

### Usage

1.  Import the module:

    ```python
    import neural_analysis_helpers as nah
    ```

2.  Use the functions:

    ```python
    frame_t, stimIDs, StimOn, StimOff, FluoData, filtered_data, filtered_pd, filtered_data_z = nah.load_data()

    nah.plot_neuronal_activity(frame_t, filtered_data_z, StimOn, StimOff)

    # ... other functions ...
    ```

## Functions

### `load_data(data_dir="Data/")`

Loads and preprocesses data from CSV and NPY files.

### `plot_neuronal_activity(time, neuronal_activity, StimOn, StimOff, neurons_to_plot=3, time_window=1000, stim_num=10)`

Plots neuronal activity with stimulus periods.

### `plot_individual_avg(filtered_data_z, avg_filt, frame_t, StimOn, StimOff, t=5000, num_neurons_to_plot=6, num_stims=40, type="sum", title='Neuronal Firing Rates with Stimulus Periods', xaxis_title='Time (s)', yaxis_title='Firing Rate (normalized)')`

Plots individual neuron firing rates and their average.

### `normalize_pcnt_act(data)`

Normalizes neuron firing rate data to percentage of maximum firing rate.

### `filter_stimuli(StimOn, StimOff, frame_t_subsection, leeway_time)`

Filters StimOn and StimOff arrays to keep only values within a specified time range.

### `plot_neuronal_activity_avg_sum(filtered_data, frame_t, StimOn, StimOff, mode='avg', time_range="full")`

Plots neuronal activity with stimulus markers, showing average or summed activity.

### `segment_data(filt_data, frame_t, StimOn, StimOff, pre_stim_samples=10, post_stim_samples=10)`

Segments neuronal data around stimulus periods.

### `segment_data_2(filt_data, frame_t, StimOn, StimOff, pre_stim_samples=5, time_to_next_stim=10)`

Segments neuronal data up to a specified time before the next stimulus.

### `plot_segmented_neuron_activity(segmented_data, max_plots=2)`

Plots individual neuron activity for segmented data.

### `segment_data_multiple_stimuli(filt_data, frame_t, StimOn, StimOff, pre_stim_samples=5, post_stim_samples=5, num_stimuli_per_segment=1)`

Segments neuronal data around multiple stimulus periods.

### `sum_segmented_data(segmented_data)`

Sums neuronal activity across neurons for each segment.

### `exponential_decay(t, a, b, c)`

Exponential decay function.

### `calculate_correlation_matrices(half1, half2, pop_avg1, pop_avg2, dt=0.032, max_lag_seconds=0.5)`

Calculates correlation matrices and correlation function matrices.

### `calculate_com(corr_matrix, lag_times)`

Calculates the center of mass (COM) for each neuron's correlation.

### `create_correlation_dataframe(corr_matrix1, corr_matrix2, exclude_self=True)`

Creates a Pandas DataFrame from flattened correlation matrices.

### `compute_corr_matrix(neural_data, pop_avg, lag_range, dt, exclude_self=False)`

Computes the cross-correlation matrix.

### `get_correlation_arrays(pop_avg, neural_data, sampling_rate=0.032, max_lag_seconds=0.5)`

Computes correlation arrays for each neuron and the population average.

### `plot_correlation_distributions(corr_df, title_suffix)`

Plots box and violin plots for correlation distributions.

### `plot_com_comparison(com1, com2, n_neurons)`

Plots a scatter plot comparing COM values between two data halves.

## Contributing

Feel free to contribute by opening issues or pull requests.

## License

This software is provided under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

**You are free to:**

* **Share:** Copy and redistribute the material in any medium or format.
* **Adapt:** Remix, transform, and build upon the material for any purpose, even commercially.

**Under the following terms:**

* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

**No additional restrictions:** You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

**Ownership:** I retain ownership of the original work.

**Link to License:** [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)