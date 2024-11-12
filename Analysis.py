# analysis.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# Reuse Spark session or create a new one
spark = SparkSession.builder.appName("Traffic Analysis").getOrCreate()

# Load data processing and fetch function from fetch.py
from fetch import fetch_traffic_data, process_data, create_spark_dataframe

# Fetch and preprocess the data
traffic_data = fetch_traffic_data()
processed_data = process_data(traffic_data)
spark_traffic_df = create_spark_dataframe(processed_data)

# --- Basic Analysis and Visualization Functions ---

# 1. Basic Summary Statistics
def summary_statistics(df):
    df.describe().show()  # Show basic statistics for all columns

# 2. Time-based Traffic Analysis (Daily Traffic Volume)
def daily_traffic_volume(df):
    # Group by 'date' and sum traffic counts
    daily_traffic = df.groupBy("date").sum(*df.columns[8:])  # Assuming traffic columns start from index 8
    daily_traffic = daily_traffic.toPandas()

    # Plot daily traffic volume
    daily_traffic.plot(x="date", y=daily_traffic.columns[1:], kind="line", figsize=(10, 6))
    plt.title("Daily Traffic Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Traffic Volume")
    plt.legend(title="Traffic Hours")
    plt.show()

# 3. Peak Hour Analysis
def peak_hour_analysis(df):
    # Calculate sum for each hour across all dates
    hour_sums = df.select([F.sum(col).alias(col) for col in df.columns[8:]])
    hour_sums.show()

    # Convert to Pandas for visualization
    hour_sums_pd = hour_sums.toPandas().T  # Transpose for plotting
    hour_sums_pd.columns = ["Traffic Volume"]

    # Plot traffic volume by hour
    hour_sums_pd.plot(kind="bar", figsize=(12, 6))
    plt.title("Traffic Volume by Hour of the Day")
    plt.xlabel("Hour")
    plt.ylabel("Total Traffic Volume")
    plt.show()

# Execute analysis functions
summary_statistics(spark_traffic_df)
daily_traffic_volume(spark_traffic_df)
peak_hour_analysis(spark_traffic_df)
