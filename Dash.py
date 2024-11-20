import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from pyspark.sql import SparkSession
import time
import requests
import os
import pandas as pd
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance


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


# Street-Based Analytics
def street_analytics(data):
    # Print dataset summary
    print("=== Dataset Summary ===")
    print(f"Total Rows: {len(data)}")
    print(f"Total Columns: {len(data.columns)}")
    print(f"Columns: {', '.join(data.columns)}")
    print("\nBasic Statistics:")
    numeric_cols = data.select_dtypes(include=['number']).columns
    print(data[numeric_cols].describe())

    # Select numeric columns for aggregation
    numeric_data = data.select_dtypes(include=['number'])

    # Group by 'roadway_name' and aggregate
    street_traffic = data.groupby('roadway_name')[numeric_data.columns].sum()

    # Reset the index to make 'roadway_name' a column again
    street_traffic.reset_index(inplace=True)

    # Identify peak traffic hours per street
    traffic_cols = [col for col in numeric_data.columns if 'am' in col.lower() or 'pm' in col.lower()]
    street_traffic['Peak Hour'] = street_traffic[traffic_cols].idxmax(axis=1)

    # Display analytics
    print("\n=== Street-Wise Traffic Volume Summary ===")
    print(street_traffic[['roadway_name', 'Peak Hour']])

    # Visualization: Total traffic by street
    street_traffic['Total Traffic'] = street_traffic[traffic_cols].sum(axis=1)
    top_streets = street_traffic.nlargest(10, 'Total Traffic')

    plt.figure(figsize=(12, 6))
    plt.bar(top_streets['roadway_name'], top_streets['Total Traffic'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Streets by Total Traffic Volume')
    plt.ylabel('Total Traffic Volume')
    plt.tight_layout()
    plt.show()


# Predictive Model
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def traffic_prediction(data):
    # Preprocessing for Date
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['day_of_week'] = data['date'].dt.dayofweek  # Numeric day of the week
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # Preprocessing for Street
    le = LabelEncoder()
    data['street_encoded'] = le.fit_transform(data['roadway_name'])

    # Add Total Traffic (as done earlier)
    traffic_cols = [col for col in data.columns if 'am' in col.lower() or 'pm' in col.lower()]
    for col in traffic_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Total Traffic'] = data[traffic_cols].sum(axis=1)

    # Features and Target
    features = ['street_encoded', 'day_of_week', 'month', 'day', 'Total Traffic']
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
    print("\n=== Traffic Prediction with XGBoost ===")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # Visualize Predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume (XGBoost)')
    plt.grid(True)
    plt.show()
    plot_importance(model) #feature importance
    plt.show()

    return model


def date_analytics(data):
    # Ensure 'date' is in datetime format
    if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Drop rows with invalid or missing dates
    data = data.dropna(subset=['date'])

    # Identify traffic-related columns
    traffic_cols = [col for col in data.columns if 'am' in col.lower() or 'pm' in col.lower()]

    # Convert traffic columns to numeric, coercing errors
    for col in traffic_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Group data by date and aggregate traffic columns
    date_traffic = data.groupby('date')[traffic_cols].sum()

    # Reset index to make 'date' a column again
    date_traffic.reset_index(inplace=True)

    # Calculate total daily traffic
    date_traffic['Total Traffic'] = date_traffic[traffic_cols].sum(axis=1)

    # Identify the busiest date
    busiest_date = date_traffic.loc[date_traffic['Total Traffic'].idxmax()]

    # Print summary
    print("\n=== Date-Wise Traffic Volume Summary ===")
    print(f"Busiest Date: {busiest_date['date']}")
    print(f"Total Traffic on Busiest Date: {busiest_date['Total Traffic']}")
    print("\nTop 5 Dates by Traffic Volume:")
    print(date_traffic.nlargest(5, 'Total Traffic'))

    # Visualization: Traffic Trend Over Time
    plt.figure(figsize=(14, 7))
    plt.plot(date_traffic['date'], date_traffic['Total Traffic'], marker='o', color='teal')
    plt.title('Traffic Volume Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Traffic Volume')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualization: Top 10 Busiest Dates
    top_dates = date_traffic.nlargest(10, 'Total Traffic')
    plt.figure(figsize=(12, 6))
    plt.bar(top_dates['date'].astype(str), top_dates['Total Traffic'], color='orange')
    plt.title('Top 10 Busiest Dates by Traffic Volume')
    plt.xlabel('Date')
    plt.ylabel('Total Traffic Volume')
    plt.xticks(rotation=45)
    plt.show()

# def analyze_errors_by_day(data, y_test_rf, y_pred_rf, y_test_xgb, y_pred_xgb):
#     # Add residuals for both models
#     data['rf_error'] = y_test_rf - y_pred_rf
#     data['xgb_error'] = y_test_xgb - y_pred_xgb
#
#     # Group by day of week and calculate mean error
#     rf_error_by_day = data.groupby('day_of_week')['rf_error'].mean()
#     xgb_error_by_day = data.groupby('day_of_week')['xgb_error'].mean()
#
#     # Plot comparison of errors for both models
#     plt.figure(figsize=(12, 6))
#     plt.plot(rf_error_by_day.index, rf_error_by_day.values, label="Random Forest", marker='o')
#     plt.plot(xgb_error_by_day.index, xgb_error_by_day.values, label="XGBoost", marker='x', color='orange')
#     plt.xlabel('Day of the Week')
#     plt.ylabel('Mean Error')
#     plt.title('Prediction Error by Day of the Week')
#     plt.legend()
#     plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
#     plt.grid(True)
#     plt.show()


if __name__ == "__main__":
    # Step 1: Fetch and process data
    traffic_data = fetch_traffic_data()
    processed_data = process_data(traffic_data)

    # Step 2: Analytics
    print("Running data analysis and visualization...")
    analyze_data(processed_data)
    plot_heatmap(processed_data)
    plot_histograms(processed_data)
    street_analytics(processed_data)
    date_analytics(processed_data)

    # Step 3: Prepare data for modeling
    print("Preparing data for modeling...")
    X, y = prepare_data_for_model(processed_data)

    # Step 4: Train and evaluate model
    print("Training and evaluating the model...")
    model, y_pred, y_test = train_model(X, y)

    # Step 5: Visualize model performance
    print("Visualizing model performance...")
    plot_model_predictions(y_test, y_pred)
    traffic_prediction(processed_data)


