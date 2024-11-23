#TrafficAnalysis.py



#Dash.py
# Traffic Analysis Dashboard

## Features

- **Real-Time Updates**: Automatically fetches and updates traffic data every 10 seconds.
- **Interactive Visualizations**:
  - Traffic volume trends by street.
  - Top streets by traffic volume.
  - Hourly traffic volume distribution.
  - Borough-wise traffic distribution (pie chart and bar chart).
  - Geographical traffic map with markers.
- **Street Selector**: Filter visualizations by selecting a specific street.


## How It Works

### Data Pipeline

- **Data Source**:  
  The app fetches traffic data from the NYC Open Data API:  
  [NYC Traffic Data API](https://data.cityofnewyork.us/resource/7ym2-wayt.json)

- **Data Fetching**:  
  - Data is fetched in chunks using pagination (`$limit`, `$offset`).  
  - Filters are applied to fetch only traffic data from the year 2024.  

- **Data Processing**:  
  - Converts fields such as `year`, `month`, `day`, `hour`, and `volume` into numeric types.  
  - Extracts latitude and longitude from geographic data (`wktgeom`).  
  - Combines date and time components into a `datetime` column for time-series analysis.  

- **Real-Time Updates**:  
  A background thread continuously fetches and processes new data.  

---

## Dashboard Components

1. **Traffic Volume Trend Line Chart**  
   Displays traffic volume trends for a selected street over time.  

2. **Top 5 Streets by Traffic Volume**  
   A bar chart showing the streets with the highest traffic volume.  

3. **Hourly Traffic Volume**  
   Bar chart showing traffic volume distribution across hours for the current day.  

4. **Borough-Wise Traffic Volume (Pie and Bar Charts)**  
   Visualizes traffic distribution across boroughs using pie and bar charts.  

5. **Traffic Volume Map**  
   An interactive map showing traffic volumes geographically with markers.  

6. **Street Selector**  
   A dropdown to select specific streets and filter the visualizations.  

---

## Code Structure

### Key Components

- **Data Fetching**:  
  - `fetch_and_process_data()` fetches, processes, and structures the data.  

- **Background Thread**:  
  - A thread continuously updates the global dataset (`global_data`).  

- **Visualization Updates**:  
  - `update_graphs()` generates visualizations dynamically based on the selected street and updated data.  

### App Layout

The app layout consists of the following Dash components:  
- `dcc.Dropdown`: For street selection.  
- `dcc.Graph`: To render charts and maps.  
- `dcc.Interval`: For automatic updates.  

---

## Known Issues and Debugging

- **Missing Data**:  
  Rows without valid latitude or longitude are dropped.  

- **Performance**:  
  Large datasets may slow down the app. Future improvements can include data sampling or caching.  

- **Error Handling**:  
  Graceful fallback is implemented for empty or erroneous data.  

---

## Future Enhancements

- **Additional Visualizations**:  
  - Traffic heatmaps for hourly trends.  
  - Weekly or monthly trend comparisons.  

- **Predictive Modeling**:  
  - Implement traffic volume predictions using machine learning.  

- **Performance Optimization**:  
  - Introduce caching mechanisms for API calls.  

- **User Features**:  
  - Allow users to filter by borough, date range, and traffic direction.  
