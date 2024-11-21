import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBRegressor

# Set up environment variables for PySpark
os.environ["PYSPARK_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Real-Time NYC Traffic Analysis") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()


# Function to fetch traffic data
def fetch_traffic_data():
    url = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
    limit = 1000  # Number of rows per request
    offset = 0  # Start at the first record
    year = 2024
    all_data = []

    while True:
        params = {
            "$limit": limit,
            "$offset": offset
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            batch = response.json()

            if not batch:  # Stop if no more data is returned
                break
            all_data.extend(batch)
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    return all_data


# Function to preprocess traffic data
def process_data(data):

    if isinstance(data, list):  # Check if the data is a list
        data = pd.DataFrame(data)

        # Convert traffic columns to numeric and handle missing values
        traffic_columns = ['volume', 'hour', 'segment_id', 'geometry']  # Example columns for simplicity
        data[traffic_columns] = data[traffic_columns].apply(pd.to_numeric, errors='coerce')
        data = data.dropna(subset=traffic_columns)  # Drop rows with missing traffic data

        # Optional: Normalize traffic data
        scaler = MinMaxScaler()
        data[traffic_columns] = scaler.fit_transform(data[traffic_columns])

    return data


# Function to create a Spark DataFrame
def create_spark_dataframe(pandas_df):
    spark_df = spark.createDataFrame(pandas_df)
    spark_df = spark_df.withColumn('volume', spark_df['volume'].cast(IntegerType()))
    return spark_df


# Data visualization: Correlation heatmap
def plot_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Traffic Data')
    plt.show()


# Prepare data for modeling
def prepare_data_for_model(data):
    # Assuming 'segment_id' and 'hour' are relevant features
    X = data[['segment_id', 'hour']]  # Features
    y = data['volume']  # Target variable: traffic volume
    return X, y


# Train Random Forest Model
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-Squared: {r2}')

    return model, y_pred, y_test


# Plot Actual vs Predicted Traffic Volume
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume')
    plt.show()


# XGBoost Traffic Prediction Model
def xgboost_traffic_prediction(X, y):
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)

    # Predictions and Evaluation
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'Mean Squared Error (XGBoost): {mse}')
    print(f'R-Squared (XGBoost): {r2}')

    plot_importance(model)  # Plot feature importance
    plt.show()

    return model

# Borough-Based Analytics
def borough_analytics(data):
    # Print dataset summary
    print("=== Borough-Wise Traffic Volume Summary ===")
    print(f"Total Rows: {len(data)}")
    print(f"Total Columns: {len(data.columns)}")
    print(f"Columns: {', '.join(data.columns)}")

    # Select numeric columns for aggregation
    numeric_data = data.select_dtypes(include=['number'])

    # Group by 'borough' and aggregate
    borough_traffic = data.groupby('borough')[numeric_data.columns].sum()

    # Reset the index to make 'borough' a column again
    borough_traffic.reset_index(inplace=True)

    # Identify peak traffic hours per borough
    traffic_cols = [col for col in numeric_data.columns if 'am' in col.lower() or 'pm' in col.lower()]
    borough_traffic['Peak Hour'] = borough_traffic[traffic_cols].idxmax(axis=1)

    # Display analytics
    print("\n=== Borough-Wise Traffic Volume Summary ===")
    print(borough_traffic[['borough', 'Peak Hour']])

    # Visualization: Total traffic by borough
    borough_traffic['Total Traffic'] = borough_traffic[traffic_cols].sum(axis=1)
    top_boroughs = borough_traffic.nlargest(10, 'Total Traffic')

    plt.figure(figsize=(12, 6))
    plt.bar(top_boroughs['borough'], top_boroughs['Total Traffic'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Boroughs by Total Traffic Volume')
    plt.ylabel('Total Traffic Volume')
    plt.tight_layout()
    plt.show()

# Predictive Analysis for Borough-Based Traffic
def borough_traffic_prediction(data):
    # Preprocessing for Date
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['day_of_week'] = data['date'].dt.dayofweek  # Numeric day of the week
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # Preprocessing for Borough
    le = LabelEncoder()
    data['borough_encoded'] = le.fit_transform(data['borough'])

    # Add Total Traffic (as done earlier)
    traffic_cols = [col for col in data.columns if 'am' in col.lower() or 'pm' in col.lower()]
    for col in traffic_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Total Traffic'] = data[traffic_cols].sum(axis=1)

    # Features and Target
    features = ['borough_encoded', 'day_of_week', 'month', 'day', 'Total Traffic']
    target = 'Total Traffic'

    # Prepare Dataset
    X = data[features].drop(columns=[target])
    y = data[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n=== Borough Traffic Prediction with XGBoost ===")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # Visualize Predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume (Borough Analysis)')
    plt.grid(True)
    plt.show()
    plot_importance(model)  # Feature importance visualization
    plt.show()

    return model


# Main function to execute the pipeline
def main():
    # Fetch and process data
    traffic_data = fetch_traffic_data()
    processed_data = process_data(traffic_data)

    # Prepare data for modeling
    X, y = prepare_data_for_model(processed_data)

    # Train and evaluate Random Forest model
    rf_model, y_pred, y_test = train_random_forest(X, y)
    plot_predictions(y_test, y_pred)

    # Train and evaluate XGBoost model
    xgb_model = xgboost_traffic_prediction(X, y)

    # Visualizations
    plot_heatmap(processed_data)  # Correlation heatmap

    # Borough analysis and visualization
    borough_analytics(processed_data)  # Perform borough-level traffic analysis
    xgb_model = borough_traffic_prediction(processed_data)  # Perform borough traffic prediction with XGBoost



if __name__ == "__main__":
    main()
