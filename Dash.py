import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import time
import threading
import requests
from pyproj import Transformer



# Function to extract latitude and longitude
def extract_lat_lon(geometry):
    try:
        coords = geometry.replace("POINT (", "").replace(")", "").split()
        if len(coords) != 2:
            return None, None
        lon, lat = map(float, coords)
        return lat, lon
    except Exception as e:
        print(f"Error parsing geometry: {geometry}, {e}")
        return None, None


# Function to fetch and process all data
def fetch_and_process_data():
    url = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
    limit = 1000
    offset = 0
    year = 2024
    data = []

    # Fetch only records from the year 2024
    while True:
        response = requests.get(url, params={"$limit": limit, "$offset": offset, "$where": f"yr={year}"})
        if response.status_code == 200:
            chunk = response.json()
            if not chunk:
                break
            data.extend(chunk)
            offset += limit
        else:
            break

    if data:
        df = pd.DataFrame(data)
        # Data processing
        df['yr'] = pd.to_numeric(df['yr'], errors='coerce')
        df['m'] = pd.to_numeric(df['m'], errors='coerce')
        df['d'] = pd.to_numeric(df['d'], errors='coerce')
        df['hh'] = pd.to_numeric(df['hh'], errors='coerce')
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce').fillna(0)
        # Extract latitude and longitude from `wktgeom`
        df['latitude'], df['longitude'] = zip(*df['wktgeom'].apply(lambda x: extract_lat_lon(x) if x else (None, None)))
        df.dropna(subset=['latitude', 'longitude'], inplace=True)  # Remove rows without coordinates

        # Combine date and time
        df['datetime'] = pd.to_datetime(df[['yr', 'm', 'd', 'hh']].rename(
            columns={'yr': 'year', 'm': 'month', 'd': 'day', 'hh': 'hour'}), errors='coerce')
        return df
    return pd.DataFrame()




# Initialize Dash app
app = dash.Dash(__name__)

# Initialize a global variable to hold the latest data
global_data = fetch_and_process_data()

# Layout of the dashboard
# Layout of the dashboard
app.layout = html.Div([
    html.H1("Real-Time NYC Traffic Dashboard (2024 Data)", style={"textAlign": "center"}),

    # Live Update Section
    dcc.Interval(
        id="interval-component",
        interval=40 * 1000,  # Update every 10 seconds
        n_intervals=0
    ),

    # Dropdown for selecting streets
    dcc.Dropdown(
        id="street-dropdown",
        options=[
            {"label": street, "value": street} for street in global_data["street"].unique()
        ] if not global_data.empty else [],
        placeholder="Select a Street",
        style={"width": "50%"}
    ),

    # Real-time traffic trend line chart
    dcc.Graph(id="traffic-line-chart"),

    # Bar chart for top streets
    dcc.Graph(id="traffic-bar-chart"),

    # Hourly traffic volume for the current date
    dcc.Graph(id="traffic-hourly-chart"),

    # Borough traffic volume pie chart
    dcc.Graph(id="traffic-boro-pie"),

    # Borough traffic volume bar chart
    dcc.Graph(id="traffic-boro-bar"),

    #map visualization
    dcc.Graph(id="traffic-map-chart")

])




# Update Data in Background
def update_global_data():
    global global_data
    while True:
        new_data = fetch_and_process_data()
        if not new_data.empty:
            global_data = new_data
        time.sleep(10)  # Fetch new data every 10 seconds



# Background thread to fetch data
thread = threading.Thread(target=update_global_data, daemon=True)
thread.start()
# Initialize transformer: from EPSG:2263 (NYC) to EPSG:4326 (WGS84)
transformer = Transformer.from_crs("epsg:2263", "epsg:4326", always_xy=True)

# Apply the transformation
global_data['longitude'], global_data['latitude'] = transformer.transform(
    global_data['longitude'].values,
    global_data['latitude'].values
)
print("Global Data Columns:", global_data.columns)
print("Sample Data:", global_data.head())
print(global_data[['latitude', 'longitude']].isnull().sum())
print(global_data[['latitude', 'longitude']].head())
print(global_data['wktgeom'].head())


# Callbacks for live updates
@app.callback(
    [
        Output("traffic-line-chart", "figure"),
        Output("traffic-bar-chart", "figure"),
        Output("traffic-hourly-chart", "figure"),
        Output("traffic-boro-pie", "figure"),
        Output("traffic-boro-bar", "figure"),
        Output("traffic-map-chart", "figure")
    ],
    [Input("street-dropdown", "value"), Input("interval-component", "n_intervals")]
)
def update_graphs(selected_street, n_intervals):
    global global_data

    # Debugging: Check if data is loaded
    if global_data.empty:
        fig_line = px.line(title="No Data Available")
        fig_bar = px.bar(title="No Data Available")
        fig_hourly = px.bar(title="No Data Available")
        fig_pie = px.pie(title="No Data Available")
        fig_boro_bar = px.bar(title="No Data Available")
        fig_map = px.scatter_mapbox(title="No Data Available")
        return fig_line, fig_bar, fig_hourly, fig_pie, fig_boro_bar, fig_map

    # Debugging: Print selected_street
    print(f"Selected Street: {selected_street}")

    # Traffic trend line chart
    if selected_street:
        filtered_data = global_data[global_data["street"] == selected_street]
        if filtered_data.empty:
            print(f"No data for selected street: {selected_street}")
            fig_line = px.line(title="No Data Available for Selected Street")
        else:
            # Reshape for time-series plot
            filtered_data["datetime"] = pd.to_datetime(filtered_data["datetime"])
            fig_line = px.line(
                filtered_data,
                x="datetime",
                y="vol",
                title=f"Traffic Volume Trend: {selected_street}"
            )
    else:
        fig_line = px.line(title="Select a Street to View Trends")

    # Bar chart for top streets
    try:
        total_volumes = global_data.groupby("street")["vol"].sum().reset_index()
        top_streets = total_volumes.nlargest(5, "vol")
        fig_bar = px.bar(
            top_streets,
            x="street",
            y="vol",
            title="Top 5 Streets by Traffic Volume"
        )
    except Exception as e:
        print(f"Error in bar chart: {e}")
        fig_bar = px.bar(title="Error in Bar Chart")

    # Hourly chart
    try:
        latest_date = global_data["datetime"].dt.date.max()
        hourly_data = global_data[global_data["datetime"].dt.date == latest_date]
        hourly_data = hourly_data.groupby(hourly_data["datetime"].dt.hour)["vol"].sum().reset_index()
        hourly_data.columns = ["hour", "traffic_volume"]
        fig_hourly = px.bar(hourly_data, x="hour", y="traffic_volume", title="Hourly Traffic Volume")
    except Exception as e:
        print(f"Error in hourly chart: {e}")
        fig_hourly = px.bar(title="Error in Hourly Chart")

    # Boro pie chart
    try:
        boro_data = global_data.groupby("boro")["vol"].sum().reset_index()
        fig_boro_pie = px.pie(boro_data, names="boro", values="vol", title="Traffic Volume by Borough")
    except Exception as e:
        print(f"Error in borough pie chart: {e}")
        fig_boro_pie = px.pie(title="Error in Borough Pie Chart")

    # Boro bar chart
    try:
        fig_boro_bar = px.bar(
            boro_data,
            x="boro",
            y="vol",
            title="Borough-wise Traffic Volume"
        )
    except Exception as e:
        print(f"Error in borough bar chart: {e}")
        fig_boro_bar = px.bar(title="Error in Borough Bar Chart")

        # Map visualization
    fig_map = px.scatter_mapbox(
        global_data,
        lat="latitude",
        lon="longitude",
        color="vol",
        size="vol",
        hover_name="street",
        hover_data=["boro", "vol", "direction"],
        title="Traffic Volume Map",
        zoom=10,
        mapbox_style="carto-positron"
    )

    return fig_line, fig_bar, fig_hourly, fig_boro_pie, fig_boro_bar, fig_map

if __name__ == "__main__":
    app.run_server(debug=True)
