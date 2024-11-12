from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
from pyspark.sql import functions as F
import requests
import os

# Set up environment variables for PySpark
os.environ["PYSPARK_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Real-Time NYC Traffic Monitoring") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()

# Define schema for traffic data based on known structure (adjust as needed)
schema = StructType([
    StructField("id", StringType(), True),
    StructField("date", TimestampType(), True),
    StructField("traffic_count", IntegerType(), True)
])

def fetch_traffic_data():
    """Fetches traffic data from NYC Open Data API"""
    url = "https://data.cityofnewyork.us/resource/btm5-ppia.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Returns a list of dictionaries
    return []

def create_spark_dataframe(data):
    """Converts fetched traffic data to a Spark DataFrame."""
    # Convert list of dictionaries to Spark DataFrame
    pandas_df = pd.DataFrame(data)
    spark_df = spark.createDataFrame(pandas_df, schema=schema)
    return spark_df

def stream_traffic_data():
    """
    Creates a streaming DataFrame that simulates continuous ingestion of traffic data.
    This uses a rate source as a placeholder for incoming real-time traffic data.
    """
    traffic_df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 1) \
        .load() \
        .selectExpr("CAST(timestamp AS TIMESTAMP) AS date", "value AS traffic_count")

    return traffic_df

# Create streaming DataFrame
streamed_df = stream_traffic_data()

# Data Transformation: Aggregating traffic data by time window for trends
traffic_trend_df = streamed_df \
    .withWatermark("date", "1 minute") \
    .groupBy(F.window(F.col("date"), "5 minutes")) \
    .agg(F.avg("traffic_count").alias("avg_traffic_volume"))

# Display data in console for inspection
query_console = traffic_trend_df.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

# Uncomment below code to send data to a TCP socket instead of console
# query_socket = traffic_trend_df.writeStream \
#     .outputMode("update") \
#     .format("socket") \
#     .option("host", "localhost") \
#     .option("port", 9999) \
#     .start()

query_console.awaitTermination()
