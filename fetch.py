from pyspark.sql import SparkSession
import time
import requests
import os
import pandas as pd
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F

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
        from sklearn.preprocessing import MinMaxScaler
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


# Fetch and process data
traffic_data = fetch_traffic_data()
processed_data = process_data(traffic_data)

# Convert pandas DataFrame to Spark DataFrame for initial inspection
spark_traffic_df = create_spark_dataframe(processed_data)

# Show initial data (this can be adjusted as per your needs)
spark_traffic_df.show(5)  # Show the first 5 rows of the Spark DataFrame

# Simulating real-time data ingestion
while True:
    data = fetch_traffic_data()
    if data:
        # Process the fetched data
        traffic_df = process_data(data)

        # Convert to Spark DataFrame
        spark_traffic_df = create_spark_dataframe(traffic_df)

        # Optionally, store or perform other operations on traffic_df
        # For example: Show the first 5 rows of the Spark DataFrame
        spark_traffic_df.show(20)

    time.sleep(10)  # Pause for 10 seconds before fetching data again
