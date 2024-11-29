
import matplotlib.pyplot as plt
import os
import re
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
import pyedflib
from matplotlib.colors import Normalize
import umap.umap_ as umap
import hdbscan
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from matplotlib.patches import Circle
from sklearn.manifold import trustworthiness
from sklearn.metrics import make_scorer



class MeaAnalyzer:
    """
    Class for analyzing Multi-Electrode Array (MEA) data.
    """

    def __init__(self, rec_dir: str = '', data_dir: str = '', fs: int = 25000, deltaT: int = 125, num_rows: int = 8, th: float = 0.01):
        """
        Initialize the MeaAnalyzer object with paths and data parameters.

        Parameters:
            rec_dir (str): Path to the recording directory.
            data_dir (str): Path to the data directory.
            fs (int): Sampling frequency in Hz.
            deltaT (int): Time interval for analysis in milliseconds.
            num_rows (int): Number of rows in the electrode array.
        """
        self.rec_dir = rec_dir
        self.data_dir = data_dir
        self.fs = fs
        self.deltaT = deltaT
        self.num_rows = num_rows
        self.th = th

    def _get_file_paths(self, directory: str, file_extension: str) -> list:
        """
        Helper function to fetch file paths with a given extension within a directory.

        Returns:
            list: Sorted list of file paths that match the file extension.
        """
        files = sorted(f for f in os.listdir(directory) if f.endswith(file_extension))
        return [os.path.join(directory, f) for f in files]

    def get_spike_csv_paths(self) -> list:
        """
        Get the paths of all spike CSV files in the data directory.

        Returns:
            list: List of spike CSV file paths.
        """
        return self._get_file_paths(self.data_dir, '.csv')

    def get_edf_file_paths(self) -> list:
        """
        Get the paths of all EDF files in the recording directory.

        Returns:
            list: List of EDF file paths.
        """
        return self._get_file_paths(self.rec_dir, '.edf')

    def load_signal_data(self, edf_file_path: str) -> tuple:
        """
        Load signal data from an EDF file.

        Parameters:
            edf_file_path (str): Path to the EDF file.

        Returns:
            tuple: Tuple containing signal data (numpy.ndarray) and channel names (list).

        """

        f = pyedflib.EdfReader(edf_file_path)
        channel_names = f.getSignalLabels()
        nchan = f.signals_in_file
        nsamp = f.getNSamples()[0]

        data = np.zeros((nchan, nsamp))
        for i in np.arange(nchan):
            data[i, :] = f.readSignal(i)
        f.close()

        return data, channel_names

    def fix2p(self, sig: np.ndarray, peaks: np.ndarray, win: int = 3) -> np.ndarray:
        """
        Adjust detected peaks based on the signal within a window around each peak.

        Parameters:
            sig (np.ndarray): Signal data.
            peaks (np.ndarray): Detected peak indices.
            win (int): Window size.

        Returns:
            np.ndarray: Array of adjusted peak indices.
        """
        peaks_fixed = []
        for p in peaks:
            s1, s2 = max(p - win, 0), min(p + win + 1, len(sig))
            local_peak = np.argmax(np.abs(sig[s1:s2])) + s1
            peaks_fixed.append(local_peak)
        return np.array(peaks_fixed)

    def fix(self, spike_df: pd.DataFrame, data: np.ndarray, channel_names: list) -> pd.DataFrame:
        """
        Correct spike data using adjusted peak detection.

         Parameters
           ----------
            spike_df (pd.DataFrame): DataFrame containing spike data.
            data (np.ndarray): Array of signal data.
            channel_names (list): List of channel names.

        Returns:
            pd.DataFrame: DataFrame with corrected spike data.
        """
        fixed_df = spike_df.copy()
        for name, group_df in spike_df.groupby('channel_name'):
            id_ch = channel_names.index(name)
            signal = data[id_ch, :]
            spikes = np.round(group_df['time'].values, decimals=3) * self.fs
            spikes = spikes.astype(int)
            fixed_df.loc[spike_df['channel_name'] == name, 'sample'] = self.fix2p(signal, spikes)

        return fixed_df

    def get_active_els(self, df: pd.DataFrame, ch_clm='channel_name', t_clm='time', time=[]):
            """
            Calculate active electrodes in a given time interval, using defined thresholds.
            """
            if len(time) != 0:
                dur = time[1] - time[0]
                df = df[df[t_clm] > time[0]]
                df = df[df[t_clm] < time[1]]
            else:
                if not df.empty:
                    dur = 900  # Assuming a default duration of 900 seconds if no time range is provided
                else:
                    dur = np.nan

            chans = list(df[ch_clm].unique())
            chans.sort()
            dfr = pd.DataFrame(columns=['channel_name', 'rate'])
            dfr_i = 0
            for chan in chans:
                count = len(df[df.channel_name == chan])
                dfr.loc[dfr_i] = [chan, count / dur]
                dfr_i += 1
            dfr = dfr[dfr.rate > self.th]

            return list(dfr.channel_name.values)

    def split_segments(self, df: pd.DataFrame)-> pd.DataFrame:
        """
           Split the dataframe into intervals and process each interval.

           Parameters
           ----------
           df : pandas DataFrame
               DataFrame containing spike data.

           Returns
           -------
           pd.DataFrame, pd.DataFrame
               Two dataframes corresponding to the two segments.
           """
        df_copy = df.copy()

        # Filter the first subset
        int1 = df_copy[df_copy['time'].between(300, 1200)]
        # Filter the second subset on the original DataFrame
        int2 = df[df['time'].between(3000, 3900)]

        segments = [int1, int2]

        el_df_seg1 = None
        el_df_seg2 = None

        for ind, interval in enumerate(segments):
            print(f"Interval {ind + 1}")
            df_interval = interval.copy()

            active_el = self.get_active_els(df_interval, ch_clm='channel_name', t_clm='time', time=[])
            el_df = df_interval[df_interval['channel_name'].isin(active_el)]
            el_df = el_df.reset_index(drop=True)
            el_df['sum'] = el_df.groupby('channel_name')['channel_name'].transform('count')

            if ind == 0:
                el_df_seg1 = el_df
            elif ind == 1:
                el_df_seg2 = el_df

        return el_df_seg1, el_df_seg2


    def spike_rate(self, df: pd.DataFrame, channel_names: list, rec_name: str) -> None:
            """
            Visualize spike rates across channels over time and display the spatial distribution of spike rates on the MEA layout.

            This method plots two types of visualizations:
            1. A time series of spike events for each channel over time, shown as a plot.
            2. An MEA layout showing the normalized spike rate per electrode, visualized as a heat map where each circle's color intensity represents the spike rate.

            Parameters:
            ----------
            df : pandas.DataFrame
                DataFrame containing spike data. This data must include 'time', 'channel_name', and 'rec_dur_us' columns.
            channel_names : list
                List of channel names that are considered in the analysis. It helps in sorting and plotting specific channel data.
            rec_name : str
                Recording name used as a label in the plot title to identify the dataset being visualized.

            Returns:
            -------
            None
                This method does not return any value but displays matplotlib plots directly.
            """
            df['time_min'] = df['time'] / 60  # Convert time to minutes
            df = df.sort_values(by='time_min')

            #del chans_non_ref[index_ref]
            plt.figure(figsize=(11, 8))
            for i, channel_name in enumerate(sorted(channel_names, reverse=True)):
                times = df[df['channel_name'] == channel_name]['time_min']
                plt.plot(times, [i] * len(times), '|', color='grey')

            plt.xlabel('Time [min]')
            plt.ylabel('Channel Name')
            plt.title(f'Spike Rate - {rec_name}')
            plt.yticks(range(len(channel_names)), sorted(channel_names, reverse=True))
            plt.xticks(range(0, int(df['time_min'].max()) + 2, 5))
            plt.ylim(-0.5, len(channel_names) - 0.5)
            plt.show()


             # Visualization of spike rate in MEA layout
            spike_counts = df.groupby('channel_name')['time'].count().reset_index().rename(columns={'time': 'spike_count'})
            time_in_sec = df['rec_dur_us'].iloc[0] / 1000000  # Convert microseconds to seconds
            spike_counts['spike_rate'] = spike_counts['spike_count'] / time_in_sec
            spike_counts['num'] = spike_counts['channel_name'].str.extract(r'E_(\d+)').astype(int)
            well_rate_mapping = dict(zip(spike_counts['num'], spike_counts['spike_rate']))

            min_spike_rate = min(spike_counts['spike_rate'])
            max_spike_rate = max(spike_counts['spike_rate'])

            layout = np.empty((self.num_rows, self.num_rows), dtype=object)
            for row in range(self.num_rows):
                for col in range(self.num_rows):
                    well_number = (col + 1) * 10 + (row + 1)
                    if well_number in well_rate_mapping:
                        spike_rate_str = '{:.4f}'.format(well_rate_mapping[well_number])
                        layout[row, col] = (f'E_{well_number}', spike_rate_str)
                    else:
                        layout[row, col] = (f'E_{well_number}', '0')

            cmap = plt.get_cmap("Reds")
            plt.figure(figsize=(10, 10))
            plt.gca().set_facecolor('white')
            cell_size = 1.0
            circle_radius = cell_size / 2.5

            for row in range(self.num_rows):
                for col in range(self.num_rows):
                    # Skip corners: assuming corners are (0,0), (0,num_rows-1), (num_rows-1,0), and (num_rows-1,num_rows-1)
                    if (row == 0 or row == self.num_rows - 1) and (col == 0 or col == self.num_rows - 1):
                        continue  # This skips the corner positions

                    if layout[row, col] is not None:
                        well_number, spike_rate = layout[row, col]
                        normalized_spike_rate = (float(spike_rate) - min_spike_rate) / (max_spike_rate - min_spike_rate)
                        circle_color = cmap(normalized_spike_rate)
                        r, g, b, _ = circle_color
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        text_color = 'white' if luminance < 0.5 else 'black'
                        circle = Circle((col * cell_size, row * cell_size), radius=circle_radius, fill=True,
                                        color=circle_color, ec='black')
                        plt.gca().add_patch(circle)
                        plt.text(col * cell_size, row * cell_size, f'{well_number}\n{spike_rate}', color=text_color,
                                 ha='center', va='center', fontsize=10)

            plt.title('60StandardMEA Layout - ' + rec_name,fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.gca().axis('equal')
            plt.gca().invert_yaxis()

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_spike_rate, vmax=max_spike_rate))
            sm.set_array([])
            cbar = plt.colorbar(sm, label='Spike Rate [spike/second]', ax=plt.gca())

            cbar.ax.yaxis.label.set_size(20)
            #plt.savefig(self.data_dir + rec_name + '_layout.png')
            plt.show()
            plt.close()

    def delay_matrix(self, spike_df: pd.DataFrame)-> np.ndarray :
        """
        Create and visualize delay matrix based on spike data.

        Parameters
        ----------
        spike_df_copy : pandas DataFrame
            DataFrame containing spike data.

        Returns
        -------
         matrix : numpy.ndarray
            Delay matrix.
        channel_mapping : dict
            Mapping from channel names to channel indices.
        """
        unique_channel_names = spike_df['channel_name'].unique()
        channel_mapping = {channel_name: index for index, channel_name in enumerate(unique_channel_names)}
        spike_df['channel_index'] = spike_df['channel_name'].map(channel_mapping)
        nchan = len(unique_channel_names)
        matrix = np.zeros((nchan, nchan))

        for i, row in spike_df.iterrows():
            spike_df_cpy = spike_df.copy()
            spike_df_cpy = spike_df_cpy[spike_df_cpy['sample'] > row['sample']]
            spike_df_cpy = spike_df_cpy[spike_df_cpy['sample'] <= row['sample'] + self.deltaT]

            if not spike_df_cpy.empty:
                spike_df_cpy = spike_df_cpy.drop_duplicates(subset='channel_name')

                for j, tmp in spike_df_cpy.iterrows():
                    if row['channel_index'] != tmp['channel_index']:
                        matrix[int(row['channel_index']), int(tmp['channel_index'])] += 1 / row['sum']

        unique_channel_names = list(channel_mapping.keys())

        # Create a larger figure
        plt.figure(figsize=(10, 8))  # Adjust the figsize as needed

        # Display the matrix using imshow with channel names as tick labels
        plt.imshow(matrix, aspect='auto', cmap='Reds')
        # Set the tick labels for the x and y axes using the channel names
        plt.xticks(np.arange(len(unique_channel_names)), unique_channel_names, rotation=90)
        plt.yticks(np.arange(len(unique_channel_names)), unique_channel_names)
        # Add a colorbar to the plot
        plt.colorbar(label="Probability of co-firing")
        # Optionally, you can set labels and a title
        plt.xlabel('Channel Name - delayed',fontsize = 20)
        plt.ylabel('Channel Name - current',fontsize = 20)
        #plt.title('Delay Map')
        #plt.savefig(delay_folder + 'int2' +'.png')
        #plt.show()
        return matrix, channel_mapping

    def delay_map(self, matrix: np.ndarray, channel_mapping: dict)-> None:
        """
        Create and visualize delay map based on delay matrix.

        Parameters
        ----------
        matrix : numpy.ndarray
            Delay matrix.
        rec_name : str
            Recording name.
        segs : list
            List containing segment information.
        channel_mapping : dict
            Mapping from channel names to channel indices.

        Returns
        -------
        None
        """
        colormap = plt.get_cmap('Reds')
        # Define layout dimensions

        # Initialize an empty layout
        layout = np.empty((self.num_rows, self.num_rows), dtype=object)

        # Generate layout based on the specified pattern
        for row in range(self.num_rows):
            for col in range(self.num_rows):
                if (row in [0, self.num_rows - 1] and col in [0, self.num_rows - 1]):
                    layout[row, col] = None
                else:
                    well_number = (col + 1) * 10 + (row + 1)
                    layout[row, col] = well_number

        # Create a plot of the layout
        plt.figure(figsize=(8, 8))
        plt.gca().set_facecolor('white')

        # Add circles with text labels for each well
        for row in range(self.num_rows):
            for col in range(self.num_rows):
                if not layout[row, col] is None:
                    circle_color = 'lightgrey'
                    text_color = 'black'
                    text = str(int(layout[row, col]))

                    if layout[row, col] == 15:
                        circle_color = 'lightcoral'
                        text_color = 'black'
                        text = f'{text}\nref'

                    circle = Circle((col, row), radius=0.4, fill=True, color=circle_color, ec='black')
                    plt.gca().add_patch(circle)
                    plt.text(col, row, text, color=text_color, ha='center', va='center', fontsize=10)

        if matrix.size > 0:
            non_zero_positions = {}
            # Create a normalization object for the colormap
            norm = Normalize(vmin=matrix.min(), vmax=matrix.max())

            # Iterate through the matrix to find non-zero elements and store their positions
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix[row, col]
                    if value > 0.25:  # != 0:
                        non_zero_positions[(row, col)] = value

            # print(non_zero_positions.keys())

            arrow_linewidth = 2.0  # Adjust this value as needed
            # Set the arrow head width
            arrow_head_width = 0.3  # Adjust this value as needed
            non_zero_colors = [colormap(norm(value)) for value in non_zero_positions.values()]
            chans_sort = []
            # Iterate through the dictionary
            for i, (key, value) in enumerate(non_zero_positions.items()):
                curr, delayed = key  # Extract row and column from the key
                # Retrieve the channel names based on curr and delayed values
                curr_channel_name = next(name for name, val in channel_mapping.items() if val == curr)
                delayed_channel_name = next(name for name, val in channel_mapping.items() if val == delayed)

                print(curr_channel_name)
                print(delayed_channel_name)
                chans_sort.append((curr_channel_name, delayed_channel_name))

                # Extract well numbers from channel names
                well_num1 = int(re.search(r'\d+', curr_channel_name).group())
                well_num2 = int(re.search(r'\d+', delayed_channel_name).group())

                # Convert well numbers to coordinates
                coord1 = [int(digit) - 1 for digit in str(well_num1)]
                coord2 = [int(digit) - 1 for digit in str(well_num2)]

                # Calculate an offset to prevent arrows from overlapping
                offset = np.array([0.1, 0.1]) * (i % 2)  # Adjust the offset as needed

                # color = colormap(value)
                color = non_zero_colors[i]

                arrow_props = dict(
                    arrowstyle='->', color=color, linewidth=arrow_linewidth,
                    shrinkA=0, shrinkB=0, mutation_scale=20, mutation_aspect=1.0,
                    connectionstyle=f"arc3,rad=0.3")
                plt.annotate('', xy=coord2 + offset, xytext=coord1 + offset, arrowprops=arrow_props)

            unique_sort = set(channel for channels_tuple in chans_sort for channel in channels_tuple)
            print(unique_sort)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])

            plt.title('Propagation map' + '-' + rec_name + '-' + segs[ind])
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.gca().axis('equal')  # Set aspect ratio to equal
            plt.gca().invert_yaxis()  # Reverse y-axis to match array indexing
            #plt.savefig(delay_folder5 + rec_name +'_'+segs[ind]+ '_propagation_map.png')
            cbar = plt.colorbar(sm, ax=plt.gca())
            # Add a colorbar to the plot
            cbar.set_label("Probability of co-firing",fontsize = 20)
            #plt.savefig(delay_folder + 'int2_m' + '.png')
            plt.show()



class SpikeSorter:
    """
    A class dedicated to the sorting of neural spikes. This class provides functionalities for initializing spike sorting parameters,
    extracting waveform segments, and preprocessing data for neural signal analysis.
    """

    def __init__(self, channel_names: list, fixed_df: pd.DataFrame, rec_name: str, deltaT: int = 125, num_rows: int = 8):
        """
        Constructs a SpikeSorter object with specified parameters for spike sorting operations.

         Parameters
            ----------
            channel_names: A list of names identifying each channel.
            fixed_df: A DataFrame containing fixed configuration data.
            rec_name: The name of the recording.
            deltaT: Time interval (in ms) around the spike event to be included in the waveform. Default is 125 ms.
            num_rows: Number of rows in the output matrix. Default is 8.
        """
        self.channel_names = channel_names
        self.fixed_df = fixed_df
        self.rec_name = rec_name
        self.deltaT = deltaT
        self.num_rows = num_rows
        self.data = data

    def extract_waveforms(self, signal: np.ndarray, spikes: np.ndarray, pre: int, post: int) -> np.ndarray:
        """
        source: https://github.com/multichannelsystems/McsPyDataTools/blob/master/McsPyDataTools/docs/McsPy-Tutorial_DataAnalysis.ipynb
        Extracts waveforms from the signal at specified spike events.

         Parameters
        ----------
             signal: The continuous signal data as a NumPy array.
             spikes: Indices of spikes within the signal as a NumPy array.
             pre: Number of samples to include before the spike event.
             post: Number of samples to include after the spike event.
        Returns
        -------
             A NumPy array of extracted waveforms.
        """
        cutouts = []
        for index in spikes:
            start_idx = max(0, index - pre)
            end_idx = min(len(signal), index + post)
            cutout = signal[start_idx:end_idx].astype(np.float32)
            cutouts.append(cutout)

        return np.stack(cutouts)

    def preprocessing(self, spike_df: pd.DataFrame, data: np.ndarray) -> tuple:
        """
        Preprocesses the spike data and constructs a matrix for spike sorting.

         Parameters
        ----------
             spike_df: DataFrame containing spike event data.
             data: The signal data as a NumPy array.

         Returns
        -------
             return: A tuple containing the spike sorting matrix, reshaped matrix, scaled cutouts, and origin and index of references.
        """
        unique_channel_names = self.fixed_df['channel_name'].unique()
        channel_mapping = {name: idx for idx, name in enumerate(unique_channel_names)}
        self.fixed_df['channel_index'] = self.fixed_df['channel_name'].map(channel_mapping)
        index_ref = self.channel_names.index("E_Ref")
        sort_mat = np.zeros((len(unique_channel_names), 2 * self.deltaT, len(self.fixed_df)))

        origin = []
        for i, row in spike_df.iterrows():
            for ch, signal in enumerate(data):
                if ch == index_ref:
                    continue
                sort_mat_index = ch if ch < index_ref else ch - 1
                spikes = np.array([row['sample']])
                cutouts = self.extract_waveforms(signal, spikes, pre=self.deltaT, post=self.deltaT)
                sort_mat[sort_mat_index, :, i] = cutouts.flatten()
            origin.append((row['channel_name'], row['sample']))

        sort_mat_2d = sort_mat.reshape(-1, sort_mat.shape[-1])
        scaler = RobustScaler()
        scaled_cutouts = scaler.fit_transform(sort_mat_2d)

        return sort_mat, sort_mat_2d.T, scaled_cutouts, origin, index_ref

    def tune_umap_parameters(self, X):
        """
        Tune UMAP parameters and return the best reducer.
         Parameters
         ----------
             X: Data to be embedded.
         Returns
         -------
               A tuple containing the best number of neighbors (`n_neigh`) and the best minimum distance (`min_dist`).
        """

        def tune_pipeline(config):
            reducer = umap.UMAP(n_neighbors=config["n_neighbors"], min_dist=config["min_dist"], n_components=2,
                                random_state=42)
            embedding = reducer.fit_transform(X)
            trustworthiness_score = trustworthiness(X, embedding)
            session.report({'trustworthiness': trustworthiness_score})

            #return reducer, trustworthiness_score

        search_space = {
            "n_neighbors": tune.randint(15, 30),  # Continuous search space for n_neighbors
            "min_dist": tune.uniform(0.00, 0.5)  # Continuous search space for min_dist
        }

        analysis = tune.run(
            tune.with_parameters(tune_pipeline, X=scaled_cutouts),
            config=search_space,
            num_samples = 150,  # Adjust as per your resources
            resources_per_trial={"cpu":96,"gpu": 8},  # Adjust as per your resources
            search_alg = OptunaSearch(metric="trustworthiness", mode="max")
        )
        best_config = analysis.get_best_config(metric="trustworthiness", mode="max")
        best_score = analysis.get_best_trial(metric="trustworthiness", mode="max").last_result["result"]
        reducer = umap.UMAP(n_neighbors=best_config["n_neighbors"], min_dist=best_config["min_dist"], n_components=2,
                            random_state=42)
        print("Best config:", best_config)
        print("Best trustworthiness score:", best_score)
        n_neigh= best_config["n_neighbors"]
        min_dist = best_config["min_dist"]

        return n_neigh,min_dist

    def hdbscan_clustering(self, scaled_cutouts: np.ndarray, n_neigh: int, min_dist: float, origin: np.ndarray):
        """
        source: https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970
        Perform HDBSCAN clustering analysis.

             Parameters
            ----------
                 scaled_cutouts (np.ndarray): Scaled spike cutouts for analysis. It should be a 2D array where each row
                                     represents a spike and each column represents a feature.
                 n_neigh (int): The number of nearest neighbors to use when computing the k-neighborhood graph.
                 min_dist (float): The minimum distance between points to be considered as connected in the clustering process.
                 origin (np.ndarray): The original data array that contains information about the spikes.

             Returns
            -------

                A dictionary containing the best parameters, DBCV score, percentage of data retained, total clusters found, and cluster sizes.
                labels: The labels assigned to each data point by the best clustering model.
                embedding: The 2D embedding of the scaled cutouts after UMAP dimensionality reduction.
                origin_with_labels: The original data array with an additional column for the cluster labels.


        """
        # Define the UMAP reducer
        reducer = umap.UMAP(n_neighbors=n_neigh, min_dist=min_dist , n_components=2, random_state=42)
        embedding = reducer.fit_transform(scaled_cutouts)

        # Run HDBSCAN clustering
        hdb = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(embedding)

        # Specify parameters and distributions to sample from
        param_dist = {
            'min_samples': [10, 20, 30, 40, 50],
            'min_cluster_size': [40, 50, 60, 100, 150],
            'cluster_selection_method': ['eom', 'leaf'],
            'metric': ['euclidean', 'manhattan']
        }

        # Define a validity scorer
        validity_scorer = make_scorer(hdbscan.validity.validity_index, greater_is_better=True)

        # Perform grid search to find the best parameters
        g_search = GridSearchCV(hdb, param_grid=param_dist, scoring=validity_scorer)
        g_search.fit(embedding)

        # Extract clustering results
        labels = g_search.best_estimator_.labels_
        clustered = (labels >= 0)

        # Calculate coverage, total clusters, and cluster sizes
        coverage = np.sum(clustered) / embedding.shape[0]
        total_clusters = np.max(labels) + 1
        cluster_sizes = np.bincount(labels[clustered]).tolist()
        origin_with_labels = np.column_stack((origin, labels))
        # Return results
        return {
            "best_parameters": g_search.best_params_,
            "DBCV_score": g_search.best_estimator_.relative_validity_,
            "percent_data_retained": coverage,
            "total_clusters_found": total_clusters,
            "cluster_splits": cluster_sizes
        }, labels, embedding,origin_with_labels




    def visualize_clusters(self,origin_with_labels: np.ndarray):
        """
           Visualizes the clusters by generating heatmaps and line plots of mean waveforms for each cluster,
           and a spatial layout of minimum amplitude values for each well.

           Parameters:
            -------
             origin_with_labels (ndarray): An array containing the origin data with cluster labels.

           Returns
            -------
            None
           """

        origin_with_labels = origin_with_labels[origin_with_labels [:, 2] != '-1']
        g_df =  pd.DataFrame(origin_with_labels)
        grouped = g_df.groupby(2)[1].apply(list).to_dict()
        time_axis = np.arange(-self.deltaT, self.deltaT)

        # Sort channel names and create an index map for reordering
        sorted_channel_names = sorted(self.channel_names)
        index_map = [self.channel_names.index(name) for name in sorted_channel_names]

        # Iterate through the groups and samples inside each group
        for group_label, samples in grouped.items():
            mean_ch_matrix = []
            for ch, signal in enumerate(self.data):
                extract = []
                for sample in samples:
                    sample = int(float(sample))  # Convert to integer index
                    if sample - self.deltaT >= 0 and sample + self.deltaT < len(data[0, :]):
                        waveform = signal[sample - self.deltaT:sample + self.deltaT]
                        waveform_detrended = waveform - np.mean(waveform)  # Detrend the waveform
                        extract.append(waveform_detrended)

                mean_ch = np.mean(extract, axis=0)
                mean_ch_matrix.append(mean_ch)

            # Convert mean_ch_matrix to a NumPy array and reorder it
            mean_ch_matrix = np.array(mean_ch_matrix)[index_map, :]

            vmin = np.min(mean_ch_matrix)
            vmax = np.max(mean_ch_matrix)

            # Dictionary to store minimum value for each channel
            min_values = {channel: np.min(mean_ch_matrix[i]) for i, channel in enumerate(sorted_channel_names)}

            # Print the minimum values for each channel
            for channel, min_value in min_values.items():
                print(f"Minimum value for {channel}: {min_value}")

            # Plotting the heatmap
            plt.figure(figsize=(10, 7))
            plt.imshow(mean_ch_matrix, aspect='auto', cmap='Reds_r', interpolation='nearest', extent = [-self.deltaT, self.deltaT, 0, len(mean_ch_matrix)])
            plt.yticks(range(len(sorted_channel_names)), sorted_channel_names[::-1])
            plt.title(f'Cluster {group_label} - Mean waveforms heatmap', fontsize=20)
            plt.xlabel('Time [samples]', fontsize=20)
            plt.ylabel('Channel names', fontsize=20)
            plt.xlim(left=-self.deltaT, right=self.deltaT)
            colorbar = plt.colorbar(label='Amplitude [uV]')
            colorbar.ax.yaxis.label.set_size(20)
            plt.xticks(np.arange(-self.deltaT, self.deltaT + 1, 10), np.arange(-self.deltaT, self.deltaT + 1, 10), fontsize=12)
            plt.tight_layout()
            #plt.savefig(f"{delay_folder}{group_label}_heatmap.png")
            #plt.show()
            plt.close()

            # Plotting the line plot
            plt.figure(figsize=(10, 7))
            for ch in range(len(mean_ch_matrix)):
                plt.plot(time_axis, mean_ch_matrix[ch], label=f'{sorted_channel_names[ch]}', color='blue')

            plt.title(f'Cluster {group_label} - Mean waveforms lineplot', fontsize=20)
            plt.xlabel('Time [samples]', fontsize=20)
            plt.ylabel('Amplitude [mV]', fontsize=20)
            plt.xlim(left=-self.deltaT, right=self.deltaT)
            plt.xticks(np.arange(-self.deltaT, self.deltaT + 1, 25), np.arange(-self.deltaT,self.deltaT + 1, 25), fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            #plt.show()
            #plt.savefig(f"{delay_folder}{group_label}_lineplot.png")
            plt.close()

            # Initialize an empty layout
            layout = np.empty((self.num_rows,self.num_rows), dtype=object)

            # Generate layout based on the specified pattern
            for row in range(self.num_rows):
                for col in range(self.num_rows):
                    if (row in [0, self.num_rows - 1] and col in [0, self.num_rows - 1]):
                        layout[row, col] = None
                    else:
                        well_number = (col + 1) * 10 + (row + 1)
                        well_number_with_prefix = 'E_' + str(well_number)  # Add "E_" prefix
                        # Check if the well_number exists in well_rate_mapping
                        if well_number_with_prefix in min_values:
                            min_ampl = min_values[well_number_with_prefix]
                            layout[row, col] = (well_number_with_prefix, min_ampl)
                        else:
                            layout[row, col] = (
                            well_number_with_prefix, 0.0)  # Set the value to 0 when well_number is not found

            # Use the "Reds" colormap
            cmap = plt.get_cmap("Reds_r")

            plt.figure(figsize=(10, 10))
            plt.gca().set_facecolor('white')

            # Calculate the circle radius based on the cell size
            cell_size = 1.0
            circle_radius = cell_size / 2.5

            # Add circles with text labels for each well
            for row in range(self.num_rows):
                for col in range(self.num_rows):
                    if layout[row, col] is not None:
                        well_number, min_ampl = layout[row, col]
                        if vmax == vmin:
                            normalized_ampl = 0.0  # or any other appropriate value
                        else:
                            normalized_ampl = (min_ampl - vmin) / (vmax - vmin)
                        circle_color = cmap(normalized_ampl)
                        # Calculate the luminance of the circle_color
                        r, g, b, _ = circle_color
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        text_color = 'white' if luminance < 0.5 else 'black'  # Choose white for dark backgrounds, black for light backgrounds
                        text = f'{well_number}'  # \nRate: {spike_rate:.6f}'
                        circle = Circle((col * cell_size, row * cell_size), radius=circle_radius, fill=True, color=circle_color,
                                        ec='black')
                        plt.gca().add_patch(circle)
                        plt.text(col * cell_size, row * cell_size, text, color=text_color, ha='center', va='center',
                                 fontsize=10)

            plt.title(f'Cluster {group_label}', fontsize=25)

            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.gca().axis('equal')  # Set aspect ratio to equal
            plt.gca().invert_yaxis()  # Reverse y-axis to match array indexing

            # Add colorbar for the "Reds" colormap
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            colorbar = plt.colorbar(sm, ax=plt.gca(), label='Amplitude [mV]')
            colorbar.ax.yaxis.label.set_size(20)
            colorbar.ax.tick_params(labelsize=20)

            #plt.savefig(delay_folder + str(group_label)+'-' + 'spatial' + '.png')
            #plt.show()
            plt.close()





if __name__ == "__main__":
    # Initialize the MeaAnalyzer with specified directories
    analyzer = MeaAnalyzer(
        rec_dir = '/path/to/edf/data/directory',
        data_dir = '/path/to/spike/detections/directory'
    )

    # Fetch paths
    spike_csv_paths = analyzer.get_spike_csv_paths()
    edf_file_paths = analyzer.get_edf_file_paths()

    if spike_csv_paths and edf_file_paths:
        idx = 4
        try:
            spike_csv_path = spike_csv_paths[idx]
            edf_file_path = edf_file_paths[idx]
            print("Processing:", spike_csv_path, edf_file_path)

            # Load and preprocess spike data
            df = pd.read_csv(spike_csv_path).dropna()
            df = df[df["channel_name"].str.contains("E_Ref") == False]
            rec_name = df['recording'].iloc[0][:-4]

            # Prepare DataFrame for processing
            spike_df = df[['channel_name', 'time', 'rec_dur_us', 'channel_index']]
            data, channel_names = analyzer.load_signal_data(edf_file_path)
            fixed_df = analyzer.fix(spike_df, data, channel_names)

            analyzer.spike_rate(fixed_df, channel_names, rec_name)
            seg1, seg2 = analyzer.split_segments(fixed_df)
            seg1_delay_matrix, channel_mapping = analyzer.delay_matrix(seg1)
            seg2_delay_matrix, channel_mapping = analyzer.delay_matrix(seg2)
            seg1_delay_map = analyzer.delay_map(seg1_delay_matrix, channel_mapping)
            seg2_delay_map = analyzer.delay_map(seg2_delay_matrix, channel_mapping)
            sorter = SpikeSorter(channel_names,fixed_df,rec_name,data)
            sort_mat, sort_mat_2d, scaled_cutouts, origin, index_ref = sorter.preprocessing(fixed_df,data,)
            n_neigh, min_dist = sorter.tune_umap_parameters(scaled_cutouts)
            results, labels_h, embedding,origin_with_labels = sorter.hdbscan_clustering(scaled_cutouts, n_neigh, min_dist,origin)
            sorter.visualize_clusters(origin_with_labels)

