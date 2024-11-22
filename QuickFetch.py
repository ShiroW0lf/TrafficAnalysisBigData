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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

    # Fetch only the first batch of 1000 rows
    params = {
        "$limit": limit,
        "$offset": offset
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        all_data = response.json()
        return all_data
    else:
        print(f"Error: {response.status_code}")
        return None



# Function to preprocess traffic data
def process_data(data):
    if isinstance(data, list):  # Check if the data is a list
        data = pd.DataFrame(data)

    # Rename columns for consistency
    data.rename(columns={
        'requestid': 'request_id', 'boro': 'borough', 'yr': 'year',
        'm': 'month', 'd': 'day', 'hh': 'hour', 'mm': 'minute',
        'vol': 'volume', 'segmentid': 'segment_id', 'wktgeom': 'geometry'
    }, inplace=True)

    # Display initial dataset info
    print("\nInitial Dataset Summary:")
    print(f"Shape of Dataset: {data.shape}")
    print("Columns and Data Types:")
    print(data.dtypes)
    print("\nPreview of the Dataset (First 5 Rows):")
    print(data.head())

    # Drop rows with missing critical fields
    required_columns = ['volume', 'hour', 'segment_id', 'borough', 'street']
    data = data.dropna(subset=required_columns)

    # Feature engineering: create additional time-related features
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])  # Combining year, month, day into a single 'date'
    data['day_of_week'] = data['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Is weekend or not
    data['week_of_year'] = data['date'].dt.isocalendar().week  # Week of the year
    data['month'] = data['date'].dt.month  # Extract month from date

    # Handle geometry (spatial data)
    if 'geometry' in data.columns:
        try:
            from shapely.wkt import loads
            data['geometry'] = data['geometry'].apply(loads)  # Convert WKT to geometry
            data['x_coord'] = data['geometry'].apply(lambda geom: geom.x if geom else None)
            data['y_coord'] = data['geometry'].apply(lambda geom: geom.y if geom else None)
        except ImportError:
            print("Shapely is not installed. Geometry features will be ignored.")
        data.drop(columns=['geometry'], inplace=True, errors='ignore')

    # Convert categorical columns to numeric only when needed for modeling (don't overwrite borough column yet)
    categorical_columns = ['direction']  # 'borough' should not be encoded here
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes

    # Convert numeric columns to numeric type
    numeric_columns = ['volume', 'hour', 'segment_id', 'year', 'month', 'day', 'minute', 'x_coord', 'y_coord']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Dataset summary after processing
    print("\nProcessed Dataset Summary:")
    print(f"Shape of Processed Dataset: {data.shape}")
    print("Columns and Data Types:")
    print(data.dtypes)
    print("\nPreview of the Processed Dataset (First 5 Rows):")
    print(data.head())
    print(data.info())
    print(data.describe())

    return data



# Function to create a Spark DataFrame
def create_spark_dataframe(pandas_df):
    spark_df = spark.createDataFrame(pandas_df)
    spark_df = spark_df.withColumn('volume', spark_df['volume'].cast(IntegerType()))
    return spark_df



# Data visualization: Correlation heatmap
def plot_heatmap(data):
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns found in the dataset for correlation heatmap.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Traffic Data')
    plt.show()


# Prepare data for modeling
def prepare_data_for_model(data):
    X = data[['segment_id', 'hour']]  # Features
    y = data['volume']  # Target variable
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




def plot_feature_importance(model, feature_names):
    # Get feature importance from the model
    feature_importances = model.feature_importances_

    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


# Train Random Forest model
def train_random_forest2(data):
    features = ['hour', 'month', 'day_of_week', 'is_weekend', 'segment_id', 'year']  # Use relevant features
    target = 'volume'

    # Prepare the data for training
    X = data[features]
    y = data[target]

    # Initialize and train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Plot feature importance
    plot_feature_importance(rf_model, features)


# Borough-wise traffic volume analysis
def analyze_boroughs(data):
    # Ensure 'borough' and 'volume' columns exist
    if 'borough' not in data.columns or 'volume' not in data.columns:
        print("Error: 'borough' or 'volume' column missing in the data.")
        return

    # Ensure 'volume' is numeric (convert if necessary)
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

    # Drop rows with NaN in 'volume' (if any)
    data = data.dropna(subset=['volume'])

    # Group by borough and calculate total volume
    borough_data = data.groupby('borough')['volume'].sum().reset_index()
    borough_data.sort_values(by='volume', ascending=False, inplace=True)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='volume', y='borough', data=borough_data, palette='viridis')
    plt.title('Total Traffic Volume by Borough')
    plt.xlabel('Total Traffic Volume')
    plt.ylabel('Borough')
    plt.show()

# Directional Traffic Volume
def analyze_directional_traffic(data):
    # Ensure 'direction' and 'volume' columns exist
    if 'direction' not in data.columns or 'volume' not in data.columns:
        print("Error: 'direction' or 'volume' column missing in the data.")
        return

    # Assuming 'direction' has been encoded to numeric values, we can map them to actual direction labels
    direction_mapping = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}  # Adjust this if needed
    data['direction_label'] = data['direction'].map(direction_mapping)

    # Group by direction and sum the traffic volume
    directional_data = data.groupby('direction_label')['volume'].sum().reset_index()

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='direction_label', y='volume', data=directional_data, palette='viridis')
    plt.title('Traffic Volume by Direction')
    plt.xlabel('Direction')
    plt.ylabel('Total Traffic Volume')
    plt.show()


def analyze_traffic_by_street(data):
    if 'street' not in data.columns:
        print("Error: 'street' column not found in the dataset.")
        return

    # Group by street and calculate total and average traffic volume
    street_traffic = data.groupby('street')['volume'].agg(['sum', 'mean']).reset_index()
    street_traffic = street_traffic.sort_values(by='sum', ascending=False).head(10)  # Top 10 busiest streets

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sum', y='street', data=street_traffic, palette='coolwarm')
    plt.title('Top 10 Streets by Total Traffic Volume')
    plt.xlabel('Total Traffic Volume')
    plt.ylabel('Street')
    plt.tight_layout()
    plt.show()

def analyze_traffic_by_date(data):
    if 'date' not in data.columns:
        print("Error: 'date' column not found in the dataset.")
        return

    # Ensure 'date' column is in datetime format
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Aggregate traffic volume by date
    date_traffic = data.groupby('date')['volume'].sum().reset_index()

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(date_traffic['date'], date_traffic['volume'], marker='o', linestyle='-', color='blue')
    plt.title('Traffic Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Traffic Volume')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Top 10 Busiest Dates by Traffic Volume
def analyze_busiest_dates(data):
    # Ensure 'date' and 'volume' columns exist
    if 'date' not in data.columns or 'volume' not in data.columns:
        print("Error: 'date' or 'volume' column missing in the data.")
        return

    # Group by date and sum the traffic volume
    busiest_dates = data.groupby('date')['volume'].sum().reset_index()

    # Sort the dates by traffic volume in descending order and pick top 10
    busiest_dates = busiest_dates.sort_values(by='volume', ascending=False).head(10)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='date', y='volume', data=busiest_dates, palette='magma')
    plt.title('Top 10 Busiest Dates by Traffic Volume')
    plt.xlabel('Date')
    plt.ylabel('Total Traffic Volume')
    plt.xticks(rotation=45)
    plt.show()


# Peak Hours Analysis
def analyze_peak_hours(data):
    # Ensure 'hour' and 'volume' columns exist
    if 'hour' not in data.columns or 'volume' not in data.columns:
        print("Error: 'hour' or 'volume' column missing in the data.")
        return

    # Group by hour and sum the traffic volume
    peak_hours = data.groupby('hour')['volume'].sum().reset_index()

    # Sort by total volume in descending order
    peak_hours = peak_hours.sort_values(by='volume', ascending=False)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y='volume', data=peak_hours, marker='o', color='blue')
    plt.title('Traffic Volume by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Traffic Volume')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.show()


def classify_traffic_conditions(data):
    # Categorize traffic volume
    bins = [0, 50, 200, np.inf]
    labels = ['Low', 'Medium', 'High']
    data['traffic_category'] = pd.cut(data['volume'], bins=bins, labels=labels, right=False)

    # Prepare features and target
    X = data[['hour', 'borough', 'segment_id', 'direction']]
    X = pd.get_dummies(X, columns=['borough', 'direction'])
    y = data['traffic_category']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    print("=== Classification Report for Traffic Conditions ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model

def classify_peak_hours(data):
    # Define peak hour label
    peak_threshold = data['volume'].quantile(0.75)  # 75th percentile
    data['is_peak_hour'] = (data['volume'] >= peak_threshold).astype(int)

    # Prepare features and target
    X = data[['hour', 'borough', 'segment_id', 'direction']]
    X = pd.get_dummies(X, columns=['borough', 'direction'])
    y = data['is_peak_hour']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    print("=== Classification Report for Peak Hours ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model

def classify_abnormal_traffic(data):
    # Define abnormal traffic label
    mean_volume = data['volume'].mean()
    std_volume = data['volume'].std()
    data['is_abnormal'] = ((data['volume'] > mean_volume + 3 * std_volume) |
                           (data['volume'] < mean_volume - 3 * std_volume)).astype(int)

    # Prepare features and target
    X = data[['hour', 'borough', 'segment_id', 'direction']]
    X = pd.get_dummies(X, columns=['borough', 'direction'])
    y = data['is_abnormal']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    print("=== Classification Report for Abnormal Traffic ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model






# Main function to execute the pipeline
def main():
    try:
        # Fetch and process data
        traffic_data = fetch_traffic_data()
        if not traffic_data:
            print("Error: No traffic data fetched. Exiting.")
            return

        processed_data = process_data(traffic_data)
        if processed_data.empty:
            print("Error: Processed data is empty. Exiting.")
            return

         # Visualizations
        plot_heatmap(processed_data)

        # Analytics: Streets and Dates
        analyze_traffic_by_street(processed_data)
        analyze_traffic_by_date(processed_data)


         #analyze
        analyze_busiest_dates(processed_data)
        analyze_peak_hours(processed_data)

        analyze_directional_traffic(processed_data)
        # Analyze boroughs
        analyze_boroughs(processed_data)



        #train data for feature importance
        train_random_forest2(processed_data)

        # Prepare data for randomforestRegression
        X, y = prepare_data_for_model(processed_data)

        # Train Random Forest model and evaluate
        model, y_pred, y_test = train_random_forest(X, y)

        # Step 5: Plot the predictions
        plot_predictions(y_test, y_pred)

        # Traffic Condition Classification
        classify_traffic_conditions(processed_data)

        # Peak Hour Classification
        classify_peak_hours(processed_data)

        # Abnormal Traffic Classification
        classify_abnormal_traffic(processed_data)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
