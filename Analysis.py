from pyspark.sql import SparkSession
import time
import sys
import os
import requests
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Real-Time NYC Traffic Analysis") \
    .getOrCreate()
print(sys.executable)


spark.sparkContext.setLogLevel("DEBUG")


def fetch_traffic_data():
    url = "https://data.cityofnewyork.us/resource/btm5-ppia.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Returns a list of dictionaries
    return []

def process_data(data):
    if isinstance(data, list):
        data = pd.DataFrame(data)
        traffic_columns = data.columns[7:]  # Assuming traffic columns start from 8th column
        for col in traffic_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.fillna(0)  # Fill NaN with 0
        return data
    return pd.DataFrame()

def create_spark_dataframe(pandas_df):
    spark_df = spark.createDataFrame(pandas_df)
    traffic_columns = spark_df.columns[7:]  # Traffic count columns start from index 7
    for col in traffic_columns:
        spark_df = spark_df.withColumn(col, spark_df[col].cast(IntegerType()))
    spark_df = spark_df.withColumn('date', F.col('date').cast('date'))
    return spark_df

# Store data locally in Parquet format
def store_locally(spark_df, path="traffic_data.parquet"):
    spark_df.write.mode("append").parquet(path)

# Fetch, process, and store data
while True:
    raw_data = fetch_traffic_data()
    if raw_data:
        processed = process_data(raw_data)
        spark_df = create_spark_dataframe(processed)
        store_locally(spark_df)  # Save to Parquet
    time.sleep(10)  # Fetch every 10 seconds

# Batch Analysis: Load stored data
stored_data = spark.read.parquet("traffic_data.parquet")

# Reshape for hourly analysis
hourly_data = stored_data \
    .select("SegmentID", "Roadway Name", "Direction", "date",
            *[F.col(c).alias(f"hour_{i}") for i, c in enumerate(stored_data.columns[8:])]) \
    .withColumn("day", F.date_format("date", "yyyy-MM-dd"))

# Summarize hourly trends
hourly_summary = hourly_data.groupBy("day").agg(
    *[F.avg(F.col(f"hour_{i}")).alias(f"avg_hour_{i}") for i in range(24)]
)

# Convert to Pandas for visualization
hourly_pd = hourly_summary.toPandas()

# Plot hourly trends
hourly_means = hourly_pd.mean(axis=0)[1:]  # Skip "day" column
plt.figure(figsize=(10, 6))
plt.plot(range(24), hourly_means, marker='o')
plt.title("Average Hourly Traffic Volume")
plt.xlabel("Hour of Day")
plt.ylabel("Traffic Volume")
plt.xticks(range(24))
plt.grid(True)
plt.show()

# Prepare data for modeling (predict traffic for specific segments/hours)
features = hourly_pd.drop(columns=["day"]).values
target = hourly_pd["avg_hour_8"].values  # Example: Predict 8 AM traffic

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
