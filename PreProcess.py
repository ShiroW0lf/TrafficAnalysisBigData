from pyspark.sql import SparkSession
import time
import requests
import os
import pandas as pd
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
import plotly.express as px

# Set up environment variables for PySpark
os.environ["PYSPARK_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"

# Initialize Spark session with HDFS configuration
spark = SparkSession.builder \
    .appName("Real-Time NYC Traffic Monitoring with HDFS") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

def fetch_traffic_data():
    url = "https://data.cityofnewyork.us/resource/btm5-ppia.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Returns a list of dictionaries
    return []

def process_data(data):
    if isinstance(data, list):
        data = pd.DataFrame(data)
        traffic_columns = data.columns[8:]  # Assuming traffic count columns start from index 8
        for col in traffic_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data[traffic_columns] = data[traffic_columns].fillna(0)
        data['date'] = data['date'].fillna(pd.to_datetime('1970-01-01'))
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data[traffic_columns] = scaler.fit_transform(data[traffic_columns])
    return data

def create_spark_dataframe(pandas_df):
    spark_df = spark.createDataFrame(pandas_df)
    traffic_columns = spark_df.columns[8:]
    for col in traffic_columns:
        spark_df = spark_df.withColumn(col, spark_df[col].cast(IntegerType()))
    spark_df = spark_df.withColumn('date', F.col('date').cast('date'))
    return spark_df

# Write streaming data to HDFS in Parquet format
def write_to_hdfs(streaming_df):
    query_hdfs = streaming_df.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "hdfs://localhost:9000/user/traffic_data") \
        .option("checkpointLocation", "hdfs://localhost:9000/user/traffic_checkpoint") \
        .start()
    return query_hdfs

# Fetch, process, and convert data to Spark DataFrame for streaming
while True:
    traffic_data = fetch_traffic_data()
    if traffic_data:
        processed_data = process_data(traffic_data)
        spark_traffic_df = create_spark_dataframe(processed_data)

        # Aggregate to calculate 5-minute average traffic volume
        traffic_trend_df = spark_traffic_df \
            .groupBy(F.window("date", "5 minutes")) \
            .agg(F.avg("traffic_volume").alias("avg_traffic_volume"))

        # Write to HDFS
        query_hdfs = write_to_hdfs(traffic_trend_df)
        query_hdfs.awaitTermination(10)  # Set a timeout for termination

    time.sleep(10)  # Pause for 10 seconds before fetching data again

# Batch analysis: load from HDFS and compute daily averages
historical_data_df = spark.read.parquet("hdfs://localhost:9000/user/traffic_data")

# Perform batch analysis for daily averages
daily_traffic_df = historical_data_df \
    .groupBy(F.date_format("window_start", "yyyy-MM-dd").alias("day")) \
    .agg(F.avg("avg_traffic_volume").alias("daily_avg_volume"))

# Collect to Pandas and visualize with Plotly
daily_traffic_pd = daily_traffic_df.toPandas()

# Plotting the daily average traffic volume
fig = px.line(daily_traffic_pd, x='day', y='daily_avg_volume', title='Daily Average Traffic Volume')
fig.show()
