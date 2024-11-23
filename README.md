# Urban Road Traffic Analysis USing Apache Spark

## Project Overview

This project aims to build a real-time traffic monitoring dashboard that tracks and analyzes traffic volume patterns across various times of day and week. The dashboard will provide insights into traffic conditions based on real-time data streams, making use of PySpark, Hadoop, and Hive for big data processing.

## Progress So Far

### 1. **Traffic Data Fetching and Processing**
   - **Data Source**: The project currently fetches traffic data from NYC's traffic monitoring system.
   - **Data Format**: The dataset includes columns for ID, segment ID, roadway name, starting point, endpoint, direction, date, and traffic volume, for each hour of the day (12:00 AM - 11:00 PM).
   - **Initial Data Processing**: 
     - The data has been successfully loaded and processed using PySpark.
     - The dataset is being filtered and transformed for analysis, particularly focusing on traffic volume by time interval.
  
### 2. **Real-Time Data Stream**
   - Set up for real-time data streaming using PySpark.
   - Currently working on establishing a continuous data ingestion process to feed the dashboard in real-time.

### 3. **Infrastructure Setup**
   - **Hadoop**: Hadoop is installed for distributed data storage and processing.
   - **Hive**: Hive is set up for querying the processed data, allowing for easy extraction of traffic trends.
   
### 4. **Planned Features**
   - **Traffic Volume Analysis**: 
     - Future analysis will involve examining daily or weekly trends in traffic volume.
   - **Real-Time Dashboard**: 
     - The goal is to visualize the traffic patterns in a user-friendly dashboard that updates in real-time.
   - **Advanced Features**: 
     - Further analysis will include detecting traffic anomalies, peak traffic times, and correlation with external factors like weather or events.

## Installation

To run the project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/traffic-monitoring-dashboard.git
   cd traffic-monitoring-dashboard
