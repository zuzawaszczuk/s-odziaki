import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

input1_dataset = pd.read_csv("data/train_115850.csv")
input2_dataset = pd.read_csv("data/empty.csv")

def get_fig(input_data=input1_dataset, output_data=input2_dataset):
    x1_dates = input_data['ds']
    x2_dates = output_data['ds']

    y1_val = input_data['yhat']
    y2_val = output_data['yhat']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter(x=x1_dates, y=y1_val))
    fig.add_trace(go.Scatter(x=x2_dates, y=y2_val))

    # Set titles
    fig.update_layout(title_text="Predicted sales")  # graph
    fig.update_xaxes(title_text="Date")  # x
    fig.update_yaxes(title_text="Item count")  # y
    fig.update_layout(showlegend=False)
    return fig


if __name__ == "__main__":
    dataset = pd.read_csv("customer_purchase_data.csv")
    dataset = dataset["date"]


#prediC(U)T+e