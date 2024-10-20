import dash
from dash import dcc, html, callback, Input, Output, callback, State, dash_table
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import base64
import datetime
import io
from src.timeseries_fig import get_fig
from src.generate_dataframe import make_predictions_prophet as make_predictions
from src.visualisation_graphs import (
    make_total_unit_sales_graph,
    make_total_unit_sales_by_city_graph,
    make_total_unit_sales_by_sales_graph,
    make_percentage_of_shops_per_city_graph,
)
from copy import deepcopy
from plotly.subplots import make_subplots

ingridient_div=html.Div()

past_df = pd.read_csv("data/empty.csv")
stores_df = pd.read_csv("data/stores.csv")

product_ingridients_data = ["Milk", "Sour", "Sugar", "Oil"]
external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?" "family=Exo:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {"background": "#111111", "text": "#7FDBFF"}
app.title = "PrediC(u)T+e"
# Define the layout of the app
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div("PrediC(u)T+e", className="title"),
                html.Div(
                    children=dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drop or ", html.A("Select .csv File")],
                            style={"textAlign": "center", "marginTop": "9px"},
                        ),
                        className="drop-box",
                        multiple=True,
                    ),
                    className="drop-box-holder",
                ),
                html.Div(
                    children=dcc.Dropdown(product_ingridients_data, id="product-data-menu"),
                    className="drop-product-menu",
                ),
            ],
            className="top-header",
        ),
        html.Div(
            children=html.Div(
                children=html.Div(
                    children=[
                        html.Div(
                            children=html.Div(id="output-data-upload"),
                            className="visualize-box",
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        dcc.Graph(
                                            id="times-series-graph",
                                            figure={},
                                            className="graph",
                                            style={"flex": 1},
                                        ),  # Start with an empty figure
                                        html.Label(
                                            "Predict x days in the future ",
                                            className="predict-label",
                                        ),
                                        dcc.Input(
                                            id="days-input",
                                            type="number",  # Use 'number' type for numeric input
                                            placeholder="x value",
                                        ),
                                    ],
                                    className="graph-box",
                                ),
                                html.Div(
                                    children=[
                                        html.Div('Suggestions', style={'fontSize':'28px', 'fontFamily':'Exo'}),
                                        html.Hr(),
                                        html.Div("Ingriedient Name:         Milk", style={'marginTop':'20px','marginBottom':'45px', 'fontSize':'22px'}),
                                        html.Div(children=[
                                            html.Div('> To Order: 3.2 t', style={'fontSize':'20px', 'flex':1, 'textAlign':'center'}),
                                            html.Div('> Model Prognose: 5.3 t', style={'fontSize':'20px', 'flex':1, 'textAlign':'center'})
                                        ],
                                        style={'display':'flex'}),
                                        html.Div([
                                            html.Div("Savings:", style={'fontSize':'20px', 'flex':1}),
                                            html.Div('20%', style={"color":'green', 'fontSize': '20px', 'flex':7})
                                        ],
                                        style={'display':'flex', 'marginTop':'60px'}
                                        )
                                    ], className="sugestions-box"
                                ),
                            ],
                            style={"width": "50%"},
                        ),
                    ],
                    className="main-box",
                ),
                className="card",
            ),
            className="wrapper",
        ),
    ]
)

# Callback to refresh data

@callback(Input(component_id="product-data-menu", component_property="value"))
def product_data_menu_div(input_value):
    product_data_menu_value = input_value


def parse_contents(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            global past_df
            past_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            #print(f"Past df after load:\n{past_df.head()}")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return [
        make_percentage_of_shops_per_city_graph(past_df, stores_df),
        make_total_unit_sales_by_sales_graph(past_df),
        make_total_unit_sales_by_city_graph(past_df, stores_df),
        make_total_unit_sales_graph(past_df),
    ]


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        return parse_contents(list_of_contents[0], list_of_names[0])


@callback(
    Output(component_id="times-series-graph", component_property="figure"),
    Input(component_id="days-input", component_property="value")
)
def plot_timeseries(time_unit=0):
    global past_df
    if time_unit is not None and time_unit <0:
        time_unit = 0
    if not past_df.empty:
        plot_part1, plot_part2 = make_predictions(deepcopy(past_df), time_unit)
        return get_fig(plot_part1, plot_part2)

    return make_subplots(specs=[[{"secondary_y": True}]])


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8000)
