# preprocessing.py

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from scipy import stats

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Define the start time of the observation period
    start_time = pd.Timestamp("2013-09-01 00:00:00")
    
    # Convert elapsed seconds to timedelta objects
    data['Time'] = pd.to_timedelta(data['Time'], unit='s')
    
    # Add elapsed seconds to the start time to get the actual datetime values
    data['Time'] = start_time + data['Time']
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    # Handle outliers using Z-score
    z_scores = stats.zscore(data.drop(['Time', 'Class'], axis=1))
    threshold = 3
    outlier_indices = np.where(np.abs(z_scores) > threshold)
    
    
    # Separate features and labels
    X = data.drop(['Class', 'Time'], axis=1)
    y = data['Class']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, X_test, y_train_smote, y_test
