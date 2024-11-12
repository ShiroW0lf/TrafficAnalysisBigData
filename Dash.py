import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import random
import time

# Initialize Dash app
app = dash.Dash(__name__)


# Dummy function to simulate real-time data
def generate_traffic_data():
    # You can replace this with real-time traffic data
    return pd.DataFrame({
        'Time': [time.strftime("%H:%M:%S")],
        'Traffic Volume': [random.randint(50, 200)]
    })


# Layout of the dashboard
app.layout = html.Div([
    html.H1("Real-Time Traffic Analysis"),
    dcc.Graph(id='traffic-graph'),
    dcc.Interval(
        id='interval-update',
        interval=2000,  # in milliseconds
        n_intervals=0
    )
])


# Callback to update the graph with new traffic data
@app.callback(
    Output('traffic-graph', 'figure'),
    Input('interval-update', 'n_intervals')
)
def update_graph(n_intervals):
    # Generate new data
    new_data = generate_traffic_data()

    # Existing data to update
    if 'df' not in globals():
        global df
        df = pd.DataFrame(columns=['Time', 'Traffic Volume'])

    # Append new data
    df = df.append(new_data, ignore_index=True)

    # Create a line chart to visualize traffic volume
    figure = {
        'data': [
            go.Scatter(
                x=df['Time'],
                y=df['Traffic Volume'],
                mode='lines+markers',
                name='Traffic Volume'
            )
        ],
        'layout': go.Layout(
            title='Traffic Volume Over Time',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Traffic Volume'},
        )
    }
    return figure


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
