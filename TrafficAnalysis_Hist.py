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
    limit = 1000  # Number of rows per request
    offset = 0    # Start at the first record
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


# Traffic analytics by Day of Week
def day_analytics(data):
    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Extract the day of the week (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    data['day_of_week'] = data['date'].dt.dayofweek

    # Group by 'day_of_week' and sum the traffic columns
    day_traffic = data.groupby('day_of_week')[traffic_cols].sum()

    # Calculate Average Traffic for each day
    day_traffic['Average Traffic'] = day_traffic.mean(axis=1)

    print(day_traffic)
    return day_traffic


def plot_day_analytics(day_traffic):
    plt.figure(figsize=(10, 6))

    # Bar chart for total traffic
    plt.bar(day_traffic.index, day_traffic['Total Traffic'], color='blue', alpha=0.7, label='Total Traffic')

    # Line chart for average traffic
    plt.plot(day_traffic.index, day_traffic['Average Traffic'], color='red', marker='o', label='Average Traffic')

    plt.xlabel('Day of the Week')
    plt.ylabel('Traffic Volume')
    plt.title('Traffic Volume by Day of the Week')
    plt.legend()
    plt.show()


# Plotting day vs. hour heatmap
def plot_day_hour_heatmap(data, traffic_cols):
    # Summing traffic for each day and hour
    day_hour_traffic = data.groupby(['day_of_week'])[traffic_cols].sum()

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(day_hour_traffic, cmap='YlGnBu', annot=True, fmt='g')
    plt.title('Traffic Volume by Day of Week and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.show()


# Define the traffic columns
traffic_cols = [
    '_12_00_1_00_am', '_1_00_2_00am', '_2_00_3_00am', '_3_00_4_00am', '_4_00_5_00am',
    '_5_00_6_00am', '_6_00_7_00am', '_7_00_8_00am', '_8_00_9_00am', '_9_00_10_00am',
    '_10_00_11_00am', '_11_00_12_00pm', '_12_00_1_00pm', '_1_00_2_00pm', '_2_00_3_00pm',
    '_3_00_4_00pm', '_4_00_5_00pm', '_5_00_6_00pm', '_6_00_7_00pm', '_7_00_8_00pm',
    '_8_00_9_00pm', '_9_00_10_00pm', '_10_00_11_00pm', '_11_00_12_00am'
]




def weekday_vs_weekend(data):
    # Create a column for weekday/weekend
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    data['is_weekend'] = ~data['day_of_week'].isin(weekdays)

    # Group by weekend status
    weekend_traffic = data.groupby('is_weekend').mean(numeric_only=True)['Total Traffic']
    weekend_traffic.index = ['Weekday', 'Weekend']

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.bar(weekend_traffic.index, weekend_traffic, color=['blue', 'orange'], alpha=0.7)
    plt.xlabel('Day Type')
    plt.ylabel('Average Traffic Volume')
    plt.title('Weekday vs Weekend Traffic Volume')
    plt.show()




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
    day_traffic = day_analytics(processed_data)
    plot_day_analytics(day_traffic)
    plot_day_hour_heatmap(processed_data, traffic_cols)
    weekday_vs_weekend(processed_data)

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


