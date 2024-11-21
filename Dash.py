import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import time
import threading
import requests

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
        interval=10 * 1000,  # Update every 10 seconds
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

# Callbacks for live updates
@app.callback(
    [
        Output("traffic-line-chart", "figure"),
        Output("traffic-bar-chart", "figure"),
        Output("traffic-hourly-chart", "figure"),
        Output("traffic-boro-pie", "figure"),
        Output("traffic-boro-bar", "figure"),
    ],
    [Input("street-dropdown", "value"), Input("interval-component", "n_intervals")]
)
def update_graphs(selected_street, n_intervals):
    global global_data

    # Ensure data is available
    if global_data.empty:
        fig_line = px.line(title="No Data Available")
        fig_bar = px.bar(title="No Data Available")
        fig_hourly = px.bar(title="No Data Available")
        fig_pie = px.pie(title="No Data Available")
        fig_boro_bar = px.bar(title="No Data Available")
        return fig_line, fig_bar, fig_hourly, fig_pie, fig_boro_bar

    # Line Chart: Traffic trend for selected street
    if selected_street:
        filtered_data = global_data[global_data["street"] == selected_street]
        fig_line = px.line(
            filtered_data,
            x="datetime", y="vol",
            title=f"Traffic Trend for {selected_street}",
            labels={"vol": "Traffic Volume", "datetime": "Time"}
        )
    else:
        fig_line = px.line(title="Select a Street to View Traffic Trend")

    # Bar Chart: Top Streets by Total Volume
    total_volume = global_data.groupby("street")["vol"].sum().reset_index()
    top_streets = total_volume.nlargest(5, "vol")
    fig_bar = px.bar(
        top_streets,
        x="street", y="vol",
        title="Top 5 Streets by Traffic Volume",
        labels={"vol": "Total Volume", "street": "Street"}
    )

    # Hourly Traffic Volume for the Current Date
    current_date = global_data["datetime"].dt.date.max()
    hourly_data = global_data[global_data["datetime"].dt.date == current_date]
    hourly_traffic = hourly_data.groupby("hh")["vol"].sum().reset_index()
    fig_hourly = px.bar(
        hourly_traffic,
        x="hh", y="vol",
        title=f"Hourly Traffic Volume for {current_date}",
        labels={"hh": "Hour", "vol": "Traffic Volume"}
    )

    # Pie Chart: Traffic Volume Distribution by Borough
    boro_volume = global_data.groupby("boro")["vol"].sum().reset_index()
    fig_pie = px.pie(
        boro_volume,
        names="boro", values="vol",
        title="Traffic Volume Distribution by Borough"
    )

    # Bar Chart: Total Traffic Volume by Borough
    fig_boro_bar = px.bar(
        boro_volume,
        x="boro", y="vol",
        title="Total Traffic Volume by Borough",
        labels={"boro": "Borough", "vol": "Total Volume"}
    )

    return fig_line, fig_bar, fig_hourly, fig_pie, fig_boro_bar

if __name__ == "__main__":
    app.run_server(debug=True)
