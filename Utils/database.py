import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
from typing import List

SEED = 3342

class Trail:
    def __init__(self, name, df, feature_columns, label_column, sample_rate):
        self.df = df
        self.name = name
        self.sample_rate = sample_rate
        self.feature_columns = feature_columns
        self.label_column = label_column

    def filter_high_freq_noise(self, sample_rate, feature_col, cutoff = 2.5):
        nyquist = 0.5 * sample_rate
        if cutoff > nyquist-1:
            print(f"cutoff {cutoff} need to be smaller than nyquist {nyquist}")
            return

        b, a = butter(N=1, Wn=cutoff / nyquist, btype='low')
        df = self.df.copy()
        df[feature_col] = filtfilt(b, a, df[feature_col].values)
        return df
    
    def get_zero_crossing_indices(self, sample_rate):
        feature_col = "Gyro_z_left"
        filtered_df = self.filter_high_freq_noise(sample_rate=sample_rate, feature_col=feature_col)
        gyro_z_values = filtered_df[feature_col].values
        zero_crossing_indices = []
        last_minima = 0
        corssing_val = -0.5
        minima_treshold = -0.7
        for index in range(1, len(gyro_z_values)-1):
            # get the last minima 
            if gyro_z_values[index-1]>gyro_z_values[index] and gyro_z_values[index]<gyro_z_values[index+1]:
                last_minima = gyro_z_values[index]

            # get a zero crossing with minima less than threshold
            if(gyro_z_values[index] < corssing_val and gyro_z_values[index+1] >= corssing_val) and last_minima < minima_treshold:
                zero_crossing_indices.append(index-5)
        return zero_crossing_indices
    
    def prepare_segs_and_labels(self, FIX_LENGTH):
        MINIMUM_GAIT_LEN = 10
        zero_crossing_indices = self.get_zero_crossing_indices(self.sample_rate)
        
        start_index = 0
        max_len = len(self.df)
        self.segIndices= []
        self.segLabels = []
        self.segs = []
        self.segGap = []

        # Iterate throught all positive zero corssing
        for index in zero_crossing_indices:
            if start_index + FIX_LENGTH >= max_len:
                break
            
            labels = self.df["FoGClass"].values[start_index: start_index+FIX_LENGTH]

            # Discard the sample if it contains FoG Episode (Too late for prediction)
            # Discard gait segments that is too short (Less than MINIMUM_GAIT_LEN)
            if 2 in labels or start_index == 0 or index - start_index <= MINIMUM_GAIT_LEN:
                start_index = index
                continue

            cur_seg_label = 0
            if 1 in labels: cur_seg_label = 1

            seg = self.df[start_index: start_index+FIX_LENGTH].copy() 
            self.segs.append(seg[self.feature_columns].values)
            self.segIndices.append(start_index)
            self.segLabels.append(cur_seg_label)

            start_index = index

        for i, cur_index in enumerate(self.segIndices):
            if i == 0:
                self.segGap.append(5.0)
                continue
            index_gap = cur_index - self.segIndices[i-1]
            time_gap = min(index_gap/self.sample_rate, 5.0)
            self.segGap.append(time_gap)

    def prepare_slidingWindow_and_labels(self, WINDOW_SIZE, WINDOW_STRIDE):
        self.slidingWindows = []
        self.slidingWindowLabels = []
        self.slidingWindowGaps = []
        self.slidingWindowIndices = []
        self.feature_columns = self.feature_columns
        for i in range(0, len(self.df)-WINDOW_SIZE, WINDOW_STRIDE):
            cur_df = self.df
            slidingWindow = cur_df[i: i+WINDOW_SIZE].copy() 
            if 2 in slidingWindow['FoGClass'].values: continue
            assert len(slidingWindow) == WINDOW_SIZE
            label = 0
            if slidingWindow['FoGClass'].values[-1] == 1:
                label = 1

            self.slidingWindows.append(slidingWindow[self.feature_columns].values)
            self.slidingWindowLabels.append(label)
            self.slidingWindowGaps.append(WINDOW_SIZE/self.sample_rate)
            self.slidingWindowIndices.append(i)

    def plot_slidingWindows(self, feature_to_plot='Gyro_z_left',num_cols = 10, skip=0):
        if not hasattr(self, 'slidingWindows') or not self.slidingWindows:
            raise Exception("NO SLIDING WINDOWS")

        num_windows = len(self.slidingWindows) - skip
        if num_windows == 0:
            print("No sliding windows to plot")
            return

        # Layout: number of columns per row and number of rows needed
        num_rows = (num_windows + num_cols - 1) // num_cols  # Ceiling division

        # If feature not present in feature_columns, show message on each subplot
        has_feature = feature_to_plot in self.feature_columns
        feature_idx = None
        if has_feature:
            feature_idx = self.feature_columns.index(feature_to_plot)

        # Create one figure per row so each row is displayed/saved separately
        for row in range(num_rows):
            fig, axes = plt.subplots(1, num_cols, figsize=(12, 2))

            # Normalize axes to a list
            if num_cols == 1:
                axes = [axes]

            for col in range(num_cols):
                idx = row * num_cols + col + skip
                ax = axes[col]

                if idx < num_windows + skip:
                    window = self.slidingWindows[idx]

                    if has_feature and feature_idx is not None:
                        feature_data = window[:, feature_idx]

                        label = self.slidingWindowLabels[idx]
                        plot_color = 'darkorange' if label == 1 else 'black'
                        ax.plot(feature_data, color=plot_color)

                        ax.set_facecolor('lightblue')
                        ax.patch.set_alpha(0.5)

                        gap = self.slidingWindowGaps[idx] if idx < len(self.slidingWindowGaps) else 0
                        ax.set_title(f'Window {idx+1}')
                        ax.set_xlabel('DP')
                
                        if col == 0:
                            ax.set_ylabel('Gyro_z (deg/s)')
                        ax.tick_params()
                        # Show only start and end ticks on x-axis for clarity
                        try:
                            ax.set_xlim(0, len(feature_data) - 1)
                            ax.set_xticks([0, len(feature_data) - 1])
                        except Exception:
                            pass
                    else:
                        ax.text(0.5, 0.5, f'Feature {feature_to_plot}\nnot found',
                                ha='center', va='center', transform=ax.transAxes)
                else:
                    # No window for this subplot in the last (partial) row
                    ax.axis('off')

            plt.tight_layout()
            plt.show()

    def plot_gaitSegments(self, feature_to_plot='Gyro_z_left',num_cols = 10, skip=0):
        if not self.segs:
            raise Exception("NO SEGS")

        num_segs = len(self.segs) - skip
        if num_segs == 0:
            print("No segments to plot")
            return

        # Layout: number of columns per row and number of rows needed
        
        num_rows = (num_segs + num_cols - 1) // num_cols  # Ceiling division

        # If feature not present in feature_columns, show message on each subplot
        has_feature = feature_to_plot in self.feature_columns
        feature_idx = None
        if has_feature:
            feature_idx = self.feature_columns.index(feature_to_plot)

        # Create one figure per row so each row is displayed/saved separately
        for row in range(num_rows):
            fig, axes = plt.subplots(1, num_cols, figsize=(12, 2))

            # Normalize axes to a list
            if num_cols == 1:
                axes = [axes]

            for col in range(num_cols):
                idx = row * num_cols + col + skip
                ax = axes[col]

                if idx < num_segs + skip:
                    seg = self.segs[idx]

                    if has_feature and feature_idx is not None:
                        feature_data = seg[:, feature_idx]

                        label = self.segLabels[idx]
                        plot_color = 'darkorange' if label == 1 else 'black'
                        ax.plot(feature_data, color=plot_color)

                        ax.set_facecolor('lightgreen')
                        ax.patch.set_alpha(0.5)

                        gap = self.segGap[idx] if idx < len(self.segGap) else 0
                        ax.set_title(f'Segment {idx+1}')
                        ax.set_xlabel('DP')
                        if col == 0:
                            ax.set_ylabel('Gyro_z (deg/s)')
                        ax.tick_params()
                        try:
                            ax.set_xlim(0, len(feature_data) - 1)
                            ax.set_xticks([0, len(feature_data) - 1])
                        except Exception:
                            pass
                    else:
                        ax.text(0.5, 0.5, f'Feature {feature_to_plot}\nnot found',
                                ha='center', va='center', transform=ax.transAxes)
                else:
                    # No segment for this subplot in the last (partial) row
                    ax.axis('off')

            plt.tight_layout()
            plt.show()

    

class Patient:
    def __init__(self, name):
        self.name = name
        self.trails: List[Trail] = []
        self.PREPARED_SEGS = False
        self.PREPARED_SW = False
        self.trail_count = 0

    def add(self, trail: Trail):
        self.trails.append(trail)
        self.trail_count = self.trail_count + 1
    

class Database:
    def __init__(self, name, feature_columns, label_column, sample_rate):
        self.REMOVED_HIGH_FREQ = False
        self.GENERATED_PREFOG = False
        self.SAMPLED_50HZ = False
        self.NORMALISED = False
        self.COMBINED_FOG = False
        self.PREPARED_SLICES = False
        self.PREPARED_SEGS = False
        self.TARGETRATE = 50
        self.name = name
        self.FILTERED_PATIENTS = False
        self.sample_rate = sample_rate
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.patients: List[Patient] = []
        self.patient_count = 0

    def add(self, patient: Patient):
        self.patients.append(patient)
        self.patient_count = self.patient_count + 1

    def normalize_features(self, method='z-score'):
        if self.NORMALISED:
            print("NORMALISED")
            return
        
        for patient in self.patients:
            for trail in patient.trails:
                df = trail.df
                for col in self.feature_columns:
                    if method == 'z-score':
                        mean = df[col].mean()
                        std = df[col].std()
                        if std != 0:  # Avoid division by zero
                            df[col] = (df[col] - mean) / std
                    elif method == 'min-max':
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val != min_val:  # Avoid division by zero
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        raise ValueError(f"Unknown normalization method: {method}")
        
        self.NORMALISED = True

    def remove_high_freq_noise(self, cutoff = 20):
        if self.REMOVED_HIGH_FREQ:
            print("REMOVED_HIGH_FREQ")
            return 
        # remove frequency component > 30Hz from the df[columns_to_process]
        nyquist = 0.5 * self.sample_rate
        if cutoff > nyquist-1:
            print(f"cutoff {cutoff} need to be smaller than nyquist {nyquist}")
            return

        b, a = butter(N=1, Wn=cutoff / nyquist, btype='low')
        for patient in self.patients:
            for trail in patient.trails:
                df = trail.df
                for col in self.feature_columns :
                    df[col] = filtfilt(b, a, df[col].values)
        self.REMOVED_HIGH_FREQ = True

    def generate_pre_fog_label(self, pre_fog_duration=1):
        if self.GENERATED_PREFOG:
            print("GENERATED_PREFOG")
            return 
        for patient in self.patients:
            for trail in patient.trails:
                df = trail.df
                # Change all FoGClass values from 1 to 2
                df.loc[df['FoGClass'] == 1, 'FoGClass'] = 2
                # Find the indices where FoGClass changes from 0 to 1 (start of FoG episode)
                fog_starts = df.index[(df['FoGClass'].shift(1) == 0) & (df['FoGClass'] == 2)].tolist()
                # Iterate through each FoG start index
                for start_idx in fog_starts:
                    # Calculate the start index for the pre_len rows before the FoG episode
                    pre_fog_start = max(0, start_idx - pre_fog_duration * self.TARGETRATE)
                    # Label the pre_len rows before the FoG episode as 1, but only if the original value is 0
                    df.loc[pre_fog_start:start_idx-1, 'FoGClass'] = df.loc[pre_fog_start:start_idx-1, 'FoGClass'].apply(lambda x: 1 if x == 0 else x)
        self.GENERATED_PREFOG = True

    def downsample_to_50hz(self):
        if self.SAMPLED_50HZ:
            print("SAMPLED_50HZ")
            return
        for patient in self.patients:
            for trail in patient.trails:
                df = trail.df
                # Create a time index in seconds
                time_index = pd.to_timedelta(np.arange(len(df)) / self.sample_rate, unit='s')
                df.index = time_index

                # Downsample columns_to_process with interpolation
                interp_cols = df[self.feature_columns].resample(f'{int(1000/self.TARGETRATE)}ms').interpolate('linear')

                # Downsample other columns with nearest
                other_cols = [col for col in df.columns if col not in self.feature_columns]
                nearest_cols = df[other_cols].resample(f'{int(1000/self.TARGETRATE)}ms').nearest()

                # Combine
                downsampled = pd.concat([interp_cols, nearest_cols], axis=1)
                downsampled = downsampled[df.columns]  # preserve original column order
                downsampled = downsampled.reset_index(drop=True)
                trail.df = downsampled
                trail.sample_rate = 50
        self.SAMPLED_50HZ = True
        self.sample_rate = 50

    def get_segs_and_labels(self, segmentLength):
#        if not self.NORMALISED: raise Exception("NOT NORMALISED")
        if not self.GENERATED_PREFOG: raise Exception("NOT GENERATED_PREFOG")
        if 'Gyro_z_left' not in self.feature_columns: raise Exception("NO Gyro_z_left in feature_columns")
        
        trails_segsSequence = []
        trails_segLabelsSequence = []
        for patient in self.patients:
            for trail in patient.trails:
                if not self.PREPARED_SEGS:
                    trail.prepare_segs_and_labels(segmentLength)
                if trail.segs:
                    trails_segsSequence.append(trail.segs)
                    trails_segLabelsSequence.append(trail.segLabels)
        self.PREPARED_SEGS = True
        return trails_segsSequence, trails_segLabelsSequence
    
    def get_slidingWindows_and_labels(self, windowSize, windowStride):
        slidingWindows = []
        slidingWindowLabels = []
        for patient in self.patients:
            for trail in patient.trails:
                if not self.PREPARED_SLICES:
                    trail.prepare_slidingWindow_and_labels(windowSize, windowStride)
                if trail.slidingWindows:
                    slidingWindows.append(trail.slidingWindows)
                    slidingWindowLabels.append(trail.slidingWindowLabels)
        
        self.PREPARED_SLICES = True
        return slidingWindows, slidingWindowLabels
    
    def plot_gaitSegments(self):
        if not self.PREPARED_SEGS:
            print("PREPARED_SEGS is missing")
            return 

        for patient in self.patients[:]:
            for trail in patient.trails:
                print(f'Plotting Gait-Cycle Based Segments for Patient ID: {patient.name}, Trail ID: {trail.name}')
                trail.plot_gaitSegments()
    
    def plot_slidingWindows(self):
        if not self.PREPARED_SLICES:
            print("PREPARED_SLICES is missing")
            return 

        for patient in self.patients[:]:
            for trail in patient.trails:
                print(f'Plotting Sliding Windows for Patient ID: {patient.name}, Trail ID: {trail.name}')
                trail.plot_slidingWindows()
