import os
import pandas as pd
from Utils.database import Database, Trail, Patient

def read_beijing_dataset(patient_no_limit=99):
    feature_columns = ['Acc_x_left', 'Acc_y_left', 'Acc_z_left',
                   'Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left']
    label_column = 'FoGClass'
    exclude_patients = ['001','002', '004', '012'] # No sensor on left leg
    sample_rate = 500 
    mydb = Database("BJ", feature_columns, label_column, sample_rate) 

    base_dir = "./Beijing Dataset/r8gmbtv7w2-FIltered/Filtered Data/"
    patients = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    count = 0
    for patient_no in patients:
        if patient_no in exclude_patients:
            continue

        count=count+1
        if(count > patient_no_limit):
            break 

        patient_path = os.path.join(base_dir, patient_no)
        trails = sorted([f for f in os.listdir(patient_path) if f.endswith('.txt')])
        tmp_patient = Patient(patient_no)
        for trail_no in trails:
            trail_path = os.path.join(patient_path, trail_no)
            df = pd.read_csv(
                trail_path,
                sep=None,
                engine='python',
                header=None,
                usecols=[1, 32, 33, 34, 35, 36, 37, 60],
                names=[
                    'Timestamp',
                    'Acc_x_left', 'Acc_y_left', 'Acc_z_left',
                    'Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left',
                    'FoGClass'
                ]
            )
            df[['Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left']] = df[['Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left']] / 1000
            # Convert timestamp strings to relative time in seconds
            timestamps = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f', errors='coerce')
            # Handle timestamps without milliseconds (like 12:04:38)
            timestamps_no_ms = pd.to_datetime(df['Timestamp'], format='%H:%M:%S', errors='coerce')
            timestamps = timestamps.fillna(timestamps_no_ms)
            
            # Convert to relative time (seconds from first timestamp)
            first_time = timestamps.iloc[0]
            df['Timestamp'] = (timestamps - first_time).dt.total_seconds()

            tmp_trail = Trail(trail_no, df, feature_columns, label_column, sample_rate)
            tmp_patient.add(tmp_trail)
        mydb.add(tmp_patient)

    return mydb


def flatten_dfs(dataframes):
    # Flatten the nested list using list comprehension
    flat_dataframes = [df for sublist in dataframes for df in sublist]
    return flat_dataframes

def concat_dfs(dataframes):
    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df