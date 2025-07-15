import pandas as pd
from sklearn.linear_model import LinearRegression
import time

def load_and_prepare_data(file_path):
    """
    Load the dataset and prepare the features (X) and target (y).
    """
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    df = df.dropna()
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 100)]
    df = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0) & (df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)]
    
    # Feature selection
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    target = 'fare_amount'
    
    X = df[features]
    y = df[target]
    
    return X, y

def benchmark_linear_regression(X, y):
    """
    Benchmark the training time of the LinearRegression model.
    """
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    training_time = end_time - start_time
    return training_time

if __name__ == "__main__":
    file_path = 'data/taxi_fare/batch1K.csv'
    
    X, y = load_and_prepare_data(file_path)
    
    training_time = benchmark_linear_regression(X, y)
    
    print(f"Scikit-learn Linear Regression training time: {training_time:.4f} seconds")
