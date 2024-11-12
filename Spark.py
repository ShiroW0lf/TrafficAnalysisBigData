from pyspark.sql import SparkSession
import os

os.environ["PYSPARK_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"  # Path to your Python executable
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\aswin\PycharmProjects\TrafficAnalysis\venv\Scripts\python.exe"







# Initialize Spark session
try:
    spark = SparkSession.builder \
        .appName("Example") \
        .config("spark.ui.port", "4050") \
        .getOrCreate()

    print("Spark session initialized successfully.")

    # Simple test DataFrame
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    df = spark.createDataFrame(data, ["Name", "Value"])
    df.show()

except Exception as e:
    print("Error initializing Spark session:", e)


