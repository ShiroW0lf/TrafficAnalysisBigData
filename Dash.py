import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from pyspark.sql import SparkSession
import time
import requests
import os
import pandas as pd
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from sklearn.preprocessing import MinMaxScaler

# Set up environment variables for PySpark
os.environ["PYSPARK_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Real-Time NYC Traffic Analysis") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()


def fetch_traffic_data():
    url = "https://data.cityofnewyork.us/resource/btm5-ppia.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Returns a list of dictionaries
    return []


def process_data(data):
    if isinstance(data, list):  # Check if the data is a list
        # Convert the list of dictionaries into a pandas DataFrame
        data = pd.DataFrame(data)

        # Perform preprocessing
        # Convert the columns representing traffic counts to numeric (since they might be strings)
        traffic_columns = data.columns[8:]  # Assuming the traffic count columns start from index 8
        for col in traffic_columns:
            # Convert to numeric, replace errors with NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Convert 'date' column to datetime type
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Handle invalid date formats

        # Fill missing values in traffic columns by interpolation or filling with 0
        # This is just an example, and should be adjusted based on the data context
        data[traffic_columns] = data[traffic_columns].fillna(0)

        # Fill missing dates with a default date or drop rows where the date is missing
        data['date'] = data['date'].fillna(pd.to_datetime('1970-01-01'))  # Replace missing dates with a default date

        # Optional: Normalize traffic data to scale the counts between 0 and 1
        # You can apply other normalization techniques as needed (like Z-score normalization)
        scaler = MinMaxScaler()
        data[traffic_columns] = scaler.fit_transform(data[traffic_columns])

        # Optional: Drop rows with too many missing values if necessary
        # For instance, if a row has more than 50% missing values, drop it
        data = data.dropna(thresh=len(data.columns) // 2)

        # Print the cleaned data for inspection
        print("Cleaned Data (first 5 rows):")
        print(data.head())

    return data


def create_spark_dataframe(pandas_df):
    # Convert pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)

    # Cast relevant columns to proper types (e.g., integers for traffic counts)
    traffic_columns = spark_df.columns[8:]  # Assuming traffic count columns are at index 8 onward
    for col in traffic_columns:
        spark_df = spark_df.withColumn(col, spark_df[col].cast(IntegerType()))

    # Cast 'date' to date type
    spark_df = spark_df.withColumn('date', F.col('date').cast('date'))

    return spark_df


# Data Analysis: Basic Statistics
def analyze_data(data):
    # Get the basic statistical summary of traffic columns
    traffic_columns = data.columns[8:]  # Assuming traffic columns start from index 8
    summary = data[traffic_columns].describe()
    print("Basic Statistics (mean, std, min, max, etc.):")
    print(summary)
    return summary


# Data Visualization: Heatmap for correlations
def plot_heatmap(data):
    traffic_columns = data.columns[8:]  # Assuming traffic columns start from index 8
    correlation_matrix = data[traffic_columns].corr()

    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Traffic Data')
    plt.show()


# Data Visualization: Histogram for traffic volume distributions
def plot_histograms(data):
    traffic_columns = data.columns[8:]  # Assuming traffic columns start from index 8
    data[traffic_columns].hist(bins=20, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Traffic Volume Distribution per Hour')
    plt.show()


# Prepare Data for Modeling
def prepare_data_for_model(data):
    # Assume that we want to predict traffic volumes based on segmentid and hour of the day
    # We can use 'segmentid' and time-based features as inputs
    traffic_columns = data.columns[8:]  # Traffic data columns
    # Flatten the data for modeling: We'll predict traffic volume at different times using 'segmentid'
    data_melted = data.melt(id_vars=['segmentid'], value_vars=traffic_columns,
                            var_name='time', value_name='traffic_volume')

    # Create time features (hour of the day, etc.)
    data_melted['hour'] = data_melted['time'].str.extract(r'(\d{1,2})_')[0].astype(int)
    data_melted['time_of_day'] = data_melted['time'].apply(lambda x: x.split('_')[1])

    # Prepare features and labels
    X = data_melted[['segmentid', 'hour']]  # Using segmentid and hour as features
    y = data_melted['traffic_volume']  # Target variable: traffic volume

    return X, y


# Build and Train Model
def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation: Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of the model: {mse}")

    return model, y_pred, y_test


# Visualize Model Predictions vs Actual
def plot_model_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume')
    plt.show()


# Fetch and process data
traffic_data = fetch_traffic_data()
processed_data = process_data(traffic_data)

# Analyze data
analyze_data(processed_data)

# Plot heatmap of correlations
plot_heatmap(processed_data)

# Plot histograms of traffic volume distribution
plot_histograms(processed_data)

# Prepare data for modeling
X, y = prepare_data_for_model(processed_data)

# Train model and evaluate
model, y_pred, y_test = train_model(X, y)

# Visualize model performance
plot_model_predictions(y_test, y_pred)
