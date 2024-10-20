import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# https://dash.plotly.com/dash-core-components/graph

def make_total_unit_sales_graph(input_data, dateformat="d-m-y"):
    input_data = input_data[input_data['unit_sales'] > 0]  

    input_data['unit_sales_7day_avg'] = input_data['unit_sales'].rolling(window=7).mean()
    input_data = input_data.dropna(subset=['unit_sales_7day_avg'])  
    
    x_dates = input_data['date']
    y_val = input_data['unit_sales']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter(x=x_dates, y=y_val, line=dict(color="#7FDBFF")))

    # Set titles
    fig.update_layout(
        title_text="Total Unit Sales",
        xaxis=dict(
            title_text="Date",
            title_standoff=40
        ),
        yaxis=dict(
            title_text="Unit Sales",
            title_standoff=40
        )
    )
    fig.update_layout(showlegend=False)
    return dcc.Graph(id="first", figure=fig, className="graph", style={'flex':1})


def make_total_unit_sales_by_city_graph(train_data, stores_data, dateformat="d-m-y"):
    train_data = train_data[train_data['unit_sales'] > 0]
    
    merged_data = pd.merge(train_data, stores_data, on='store_nbr')
    sales_per_city = merged_data.groupby('city')['unit_sales'].sum().sort_values(ascending=False)
    x_cities = sales_per_city.index.values
    y_val = sales_per_city.values

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Bar(x=x_cities, y=y_val, marker=dict(color="#7FDBFF")))

    # Set titles
    fig.update_layout(
        title_text="Total Unit Sales by City",
        xaxis=dict(
            title_text="City",
            title_standoff=40
        ),
        yaxis=dict(
            title_text="Unit Sales",
            title_standoff=40
        )
    )
    fig.update_layout(showlegend=False)
    return dcc.Graph(id="second", figure=fig, className="graph", style={'flex':1})

def make_percentage_of_shops_per_city_graph(train_data, stores_data, dateformat="d-m-y"):
    train_data = train_data[train_data['unit_sales'] > 0]
    
    merged_data = pd.merge(train_data, stores_data, on='store_nbr')
    shops_per_city = merged_data.groupby('city')['store_nbr'].count().sort_values(ascending=False)
    
    x_shops = shops_per_city.index.values
    y_val = shops_per_city.values

    fig = go.Figure(data=[go.Pie(labels=x_shops, values=y_val, hole=0.3)])
    fig.update_traces(textinfo='label')

    # Set titles
    fig.update_layout(title_text="Percentage of Shops per City")  # graph
    fig.update_layout(showlegend=False)
    return dcc.Graph(id="second", figure=fig, className="graph", style={'flex':1})

def make_total_unit_sales_by_sales_graph(train_data, dateformat="d-m-y"):
    train_data = train_data[train_data['unit_sales'] > 0]
    sales_per_sale = train_data.groupby('unit_sales')['unit_sales'].count().sort_values(ascending=False)
    sales_per_sale = sales_per_sale[sales_per_sale.index < 40]
    x_sales = sales_per_sale.index.values
    y_val = sales_per_sale.values

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Bar(x=x_sales, y=y_val, marker=dict(color="#7FDBFF")))

    # Set titles
    fig.update_layout(
        title_text="Total Unit Sales by Sales Amount",
        xaxis=dict(
            title_text="Unit Sales Amount",
            title_standoff=40
        ),
        yaxis=dict(
            title_text="Count of Occurances",
            title_standoff=40
        )
    )
    fig.update_layout(showlegend=False)
    return dcc.Graph(id="second", figure=fig, className="graph", style={'flex':1})