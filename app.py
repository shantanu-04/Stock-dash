from model import fetch_stock_prices, train_svr_model
import numpy as np
import dash
from dash import dcc
from dash import html
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

app.css.append_css({'external_url':'styles.css'})

app.layout = html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),

        html.Div([
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Get Company Info', id='company-info-button'),
        ], className='input-section'),

        html.Div([
            dcc.DatePickerRange(
                id='date-range-picker',
                display_format='YYYY-MM-DD',
                start_date='2023-01-01',
                end_date='2024-01-01',
            ),
            html.Button('Get Stock Price', id='stock-price-button'),
        ], className='input-section'),

        html.Div([
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days for forecast'),
            html.Button('Get Indicators', id='indicators-button'),
            html.Button('Get Forecast', id='forecast-button'),
        ], className='input-section'),

        html.Div(
            [
                html.Img(id='company-logo'),
                html.H3(id='company-name'),
            ],
            className="header"
        ),

        html.Div(
            [
                html.Div(id='description', className="description-ticker"),
                dcc.Graph(id='stock-price-plot'),
            ],
            id="graphs-content"
        ),

        html.Div(
            [
                dcc.Graph(id='indicator-plot'),
            ],
            id="main-content"
        ),

        html.Div(
            [
                dcc.Graph(id='forecast-plot'),
            ],
            id="forecast-content"
        )

    ],
    className="nav"
)

# First callback function for fetching company info
@app.callback(
    [
        Output('company-name', 'children'),
        Output('company-logo', 'src'),
        Output('description', 'children')
    ],
    [Input('company-info-button', 'n_clicks')],
    [State('stock-code', 'value')]
)
def update_company_info(n_clicks, stock_code):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update

    try:
        ticker = yf.Ticker(stock_code)
        info = ticker.info

        company_name = info.get('shortName', 'N/A')
        logo_url = info.get('logo_url', '')
        description = info.get('longBusinessSummary', 'No description available.')

        return company_name, logo_url, description
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return dash.no_update, dash.no_update, dash.no_update


# Second callback function for fetching stock price history and plotting
@app.callback(
    Output('stock-price-plot', 'figure'),
    [Input('stock-price-button', 'n_clicks')],
    [
        State('stock-code', 'value'),
        State('date-range-picker', 'start_date'),
        State('date-range-picker', 'end_date')
    ]
)
def update_stock_price_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks is None:
        return dash.no_update

    try:
        df = yf.download(stock_code, start=start_date, end=end_date)
        df.reset_index(inplace=True)

        # Create a simple line plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.update_layout(title="Stock Price vs Date", xaxis_title='Date', yaxis_title='Price')

        return fig
    except Exception as e:
        print(f"Error fetching stock price: {e}")
        return dash.no_update

# Third callback function for generating indicator plot
@app.callback(
    Output('indicator-plot', 'figure'),
    [Input('indicators-button', 'n_clicks')],
    [
        State('stock-code', 'value'),
        State('date-range-picker', 'start_date'),
        State('date-range-picker', 'end_date')
    ]
)
def update_indicator_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks is None:
        return dash.no_update

    try:
        df = yf.download(stock_code, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        fig = get_ema_fig(df)
        return fig
    except Exception as e:
        print(f"Error generating indicator plot: {e}")
        return dash.no_update

# User-defined function for EMA plot
def get_ema_fig(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x='Date', y='EWA_20', title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

# Callback function for stock prediction
@app.callback(
    Output('forecast-plot', 'figure'),
    [Input('forecast-button', 'n_clicks')],
    [
        State('stock-code', 'value'),
        State('forecast-days', 'value')
    ]
)
def predict_stock_price(n_clicks, stock_code, forecast_days):
    if n_clicks is None:
        return dash.no_update

    try:
        # Fetch stock prices for the last 60 days
        stock_prices = fetch_stock_prices(stock_code)

        # Train the SVR model and get predictions for the next 'forecast_days'
        svr_model, _, _ = train_svr_model(stock_prices.ravel())
        future_dates = np.arange(len(stock_prices), len(stock_prices) + forecast_days).reshape(-1, 1)
        forecasted_prices = svr_model.predict(future_dates)

        # Print values for debugging
        print(f"Stock Prices: {stock_prices}")
        print(f"Forecasted Prices: {forecasted_prices}")

        # Check for NaN values
        if np.isnan(forecasted_prices).any():
            raise ValueError("Forecasted prices contain NaN values.")

        # Create a simple line plot for the forecasted prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(stock_prices)), y=stock_prices, mode='lines', name='Actual Prices'))
        fig.add_trace(go.Scatter(x=future_dates.flatten(), y=forecasted_prices, mode='lines', name='Forecasted Prices'))

        fig.update_layout(title="Stock Price Forecast", xaxis_title='Days', yaxis_title='Price')
        return fig

    except Exception as e:
        print(f"Error predicting stock price: {e}")
        return dash.no_update



if __name__ == '__main__':
    app.run_server(debug=True)
