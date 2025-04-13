import streamlit as st  # Web app framework for interactive dashboards
import yfinance as yf  # Library for fetching stock data from Yahoo Finance
import pandas as pd  # Data manipulation and analysis library
import plotly.graph_objects as go  # Plotly module for creating interactive plots
from plotly.subplots import make_subplots  # Helper for creating subplot layouts in Plotly
import tempfile  # Module for creating temporary files
import os  # OS module for interacting with the operating system
import json  # Library for handling JSON data
from datetime import datetime, timedelta  # For working with dates and time intervals
from io import BytesIO  # In-memory binary streams for image handling
import re  # Regular expressions for text processing
import base64  # For encoding binary data into a base64 string for downloads

# ta library imports for technical analysis indicators
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, MFIIndicator

# Import our AI model functions
from model import generate_technical_analysis
from transformers import pipeline  
import torch
torch.classes.__path__ = []
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------------------------------------------------------------------
# Streamlit Setup
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide")  # Configure the Streamlit app layout to use the full width of the page
st.title("Multimodal Technical Analysis")  # Set the main title of the dashboard
st.sidebar.header("Configuration")  # Add a header in the sidebar for configuration options

# Reset Button to clear session state if needed
if st.sidebar.button("Reset App"):
    st.session_state.clear()

# ------------------------------------------------------------------------------
# Timeframe & Date Range
# ------------------------------------------------------------------------------
# Sidebar selectbox to choose the chart timeframe
timeframe = st.sidebar.selectbox(
    "Timeframe:",
    ["Daily", "Weekly", "1 Hour", "30 Minutes", "15 Minutes", "5 Minutes", "1 Minute"],
    index=0
)
# Map human-readable timeframes to yfinance interval codes
interval_map = {
    "Daily": "1d",
    "Weekly": "1wk",
    "1 Hour": "1h",
    "30 Minutes": "30m",
    "15 Minutes": "15m",
    "5 Minutes": "5m",
    "1 Minute": "1m"
}
yf_interval = interval_map[timeframe]  # Determine the interval to use for fetching data

end_date_default = datetime.today()  # Default end date is today's date
# Set a shorter historical period for intraday data and a longer one for daily/weekly data
if timeframe in ["1 Hour", "30 Minutes", "15 Minutes", "5 Minutes", "1 Minute"]:
    start_date_default = end_date_default - timedelta(days=7)
else:
    start_date_default = end_date_default - timedelta(days=365)

# Date inputs in the sidebar for selecting custom start and end dates
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# ------------------------------------------------------------------------------
# Ticker & Indicator Selection
# ------------------------------------------------------------------------------
# Sidebar text input for users to enter stock ticker symbols (comma-separated)
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "NVDA, TSM")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]  # Clean and format ticker symbols

# Sidebar multi-select for choosing technical indicators to display on the chart
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    [
        "SMA", "EMA", "Bollinger Bands", "VWAP", "Fibonacci Retracements",
        "RSI", "MACD", "OBV", "Stochastic Oscillator", "ATR", "ADX",
        "CCI", "Williams %R", "ROC", "MFI"
    ],
    default=["SMA"]
)

selected_indicators_code = indicators  # Store selected indicators for later reference

# Set of indicators that will be overlaid directly on the main price chart
overlay_indicators_set = {"SMA", "EMA", "Bollinger Bands", "VWAP", "Fibonacci Retracements"}

# ------------------------------------------------------------------------------
# Indicator Parameters
# ------------------------------------------------------------------------------
indicator_params = {}
# Provide sliders for indicator-specific parameters in the sidebar based on selected indicators
if any(i in indicators for i in ["SMA", "EMA", "Bollinger Bands"]):
    indicator_params["length_20"] = st.sidebar.slider("Length (SMA/EMA/Bollinger)", 5, 100, 20)
if "RSI" in indicators:
    indicator_params["rsi_length"] = st.sidebar.slider("RSI Length", 5, 30, 14)
if "MACD" in indicators:
    indicator_params["macd_fast"] = st.sidebar.slider("MACD Fast Length", 5, 30, 12)
    indicator_params["macd_slow"] = st.sidebar.slider("MACD Slow Length", 10, 50, 26)
    indicator_params["macd_signal"] = st.sidebar.slider("MACD Signal Length", 3, 20, 9)
if "Stochastic Oscillator" in indicators:
    indicator_params["stoch_k"] = st.sidebar.slider("Stoch %K Length", 5, 30, 14)
    indicator_params["stoch_d"] = st.sidebar.slider("Stoch %D Smoothing", 2, 10, 3)
    indicator_params["stoch_smooth_k"] = st.sidebar.slider("Stoch %K Smoothing", 2, 10, 3)
if "ATR" in indicators:
    indicator_params["atr_length"] = st.sidebar.slider("ATR Length", 5, 30, 14)
if "ADX" in indicators:
    indicator_params["adx_length"] = st.sidebar.slider("ADX Length", 5, 30, 14)
if "CCI" in indicators:
    indicator_params["cci_length"] = st.sidebar.slider("CCI Length", 5, 30, 20)
if "Williams %R" in indicators:
    indicator_params["wr_length"] = st.sidebar.slider("Williams %R Length", 5, 30, 14)
if "ROC" in indicators:
    indicator_params["roc_length"] = st.sidebar.slider("ROC Length", 5, 30, 12)
if "MFI" in indicators:
    indicator_params["mfi_length"] = st.sidebar.slider("MFI Length", 5, 14, 14)

# ------------------------------------------------------------------------------
# The Analyze Function
# ------------------------------------------------------------------------------
def analyze_ticker(ticker, data, indicator_params, start_date, end_date):
    """
    Analyzes a given stock ticker's historical data, overlays technical indicators,
    and constructs a multi-panel Plotly chart.

    Parameters:
        ticker (str): Stock ticker symbol.
        data (DataFrame): Historical price and volume data.
        indicator_params (dict): Parameters for the technical indicators.
        start_date (date): Analysis start date.
        end_date (date): Analysis end date.

    Returns:
        tuple: A Plotly figure and an optional error analysis object.
    """
    try:
        # Separate oscillator indicators from overlay indicators for layout purposes
        oscillator_list = [ind for ind in indicators if ind not in overlay_indicators_set]
        total_rows = 2 + len(oscillator_list)  # 1 row for candlestick, 1 for volume, rest for oscillators

        row_specs = []  # Specifications for each subplot row
        row_heights = []  # Relative heights for each subplot row

        # Main price chart (candlestick)
        row_specs.append([{"secondary_y": False}])
        row_heights.append(0.4)
        # Volume bar chart row
        row_specs.append([{"secondary_y": False}])
        row_heights.append(0.2)

        # Additional rows for oscillator indicators
        if oscillator_list:
            each_height = 0.4 / len(oscillator_list)
            for _ in oscillator_list:
                row_specs.append([{"secondary_y": False}])
                row_heights.append(each_height)

        # Create a Plotly figure with subplots for price, volume, and oscillators
        fig = make_subplots(
            rows=total_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=row_heights,
            specs=row_specs
        )

        # Add candlestick chart trace for price data
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            ),
            row=1, col=1
        )

        # Add volume bar chart trace
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color='rgba(166, 166, 166, 0.5)'
            ),
            row=2, col=1
        )

        # Hide x-axis tick labels for all rows except the bottom one
        for r in range(1, total_rows):
            if r != total_rows:
                fig.update_xaxes(showticklabels=False, row=r, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)  # Remove the default range slider

        # Nested function to add overlay indicators on the main price chart
        def add_overlay(ind_name):
            length_20  = indicator_params.get("length_20", 20)
            if ind_name == "SMA":
                # Calculate Simple Moving Average and add to chart
                sma_val = ta.trend.SMAIndicator(close=data['Close'], window=length_20).sma_indicator()
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_val, mode='lines', name=f"SMA({length_20})"),
                    row=1, col=1
                )
            elif ind_name == "EMA":
                # Calculate Exponential Moving Average and add to chart
                ema_val = ta.trend.EMAIndicator(close=data['Close'], window=length_20).ema_indicator()
                fig.add_trace(
                    go.Scatter(x=data.index, y=ema_val, mode='lines', name=f"EMA({length_20})"),
                    row=1, col=1
                )
            elif ind_name == "Bollinger Bands":
                # Compute Bollinger Bands and add upper and lower bands to chart
                bb = BollingerBands(close=data['Close'], window=length_20, window_dev=2)
                upper = bb.bollinger_hband()
                lower = bb.bollinger_lband()
                fig.add_trace(
                    go.Scatter(x=data.index, y=upper, mode='lines', name=f"BB Upper({length_20})", line=dict(dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=lower, mode='lines', name=f"BB Lower({length_20})", line=dict(dash='dash')),
                    row=1, col=1
                )
            elif ind_name == "VWAP":
                # Compute Volume Weighted Average Price and overlay it
                vwap_data = ta.volume.VolumeWeightedAveragePrice(
                    high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']
                ).volume_weighted_average_price()
                fig.add_trace(
                    go.Scatter(x=data.index, y=vwap_data, mode='lines', name="VWAP"),
                    row=1, col=1
                )
            elif ind_name == "Fibonacci Retracements":
                # Compute key Fibonacci retracement levels based on high and low price points
                high_pt = data['High'].max()
                low_pt  = data['Low'].min()
                diff = high_pt - low_pt
                fibs = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                for f in fibs:
                    lvl_price = high_pt - diff*f
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index[0], data.index[-1]],
                            y=[lvl_price, lvl_price],
                            mode='lines',
                            line=dict(dash='dot'),
                            name=f"Fib {f*100:.1f}%"
                        ),
                        row=1, col=1
                    )

        # Nested function to add oscillator indicators to designated subplot rows
        def add_oscillator(ind_name, row_idx):
            # Retrieve parameter values with defaults if not specified
            length_20  = indicator_params.get("length_20", 20)
            rsi_length = indicator_params.get("rsi_length", 14)
            macd_fast  = indicator_params.get("macd_fast", 12)
            macd_slow  = indicator_params.get("macd_slow", 26)
            macd_signal= indicator_params.get("macd_signal", 9)
            stoch_k    = indicator_params.get("stoch_k", 14)
            stoch_d    = indicator_params.get("stoch_d", 3)
            atr_length = indicator_params.get("atr_length", 14)
            adx_length = indicator_params.get("adx_length", 14)
            cci_length = indicator_params.get("cci_length", 20)
            wr_length  = indicator_params.get("wr_length", 14)
            roc_length = indicator_params.get("roc_length", 12)
            mfi_length = indicator_params.get("mfi_length", 14)

            if ind_name == "RSI":
                # Compute Relative Strength Index
                rsi_val = ta.momentum.RSIIndicator(close=data['Close'], window=rsi_length).rsi()
                fig.add_trace(
                    go.Scatter(x=data.index, y=rsi_val, mode='lines', name=f"RSI({rsi_length})"),
                    row=row_idx, col=1
                )
            elif ind_name == "MACD":
                # Compute MACD components: line, signal, and histogram
                macd_obj = ta.trend.MACD(
                    close=data['Close'],
                    window_slow=macd_slow,
                    window_fast=macd_fast,
                    window_sign=macd_signal
                )
                macd_line = macd_obj.macd()
                macd_sig  = macd_obj.macd_signal()
                macd_hist = macd_obj.macd_diff()
                fig.add_trace(
                    go.Scatter(x=data.index, y=macd_line, mode='lines', name="MACD Line"),
                    row=row_idx, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=macd_sig, mode='lines', line=dict(dash='dash'), name="MACD Signal"),
                    row=row_idx, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=macd_hist, mode='lines', line=dict(dash='dot'), name="MACD Hist"),
                    row=row_idx, col=1
                )
            elif ind_name == "OBV":
                # Compute On Balance Volume
                obv_val = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
                fig.add_trace(
                    go.Scatter(x=data.index, y=obv_val, mode='lines', name="OBV"),
                    row=row_idx, col=1
                )
            elif ind_name == "Stochastic Oscillator":
                # Compute the Stochastic Oscillator %K and %D lines
                stoch = ta.momentum.StochasticOscillator(
                    high=data['High'], low=data['Low'], close=data['Close'],
                    window=stoch_k, smooth_window=stoch_d
                )
                k_val = stoch.stoch()
                d_val = stoch.stoch_signal()
                fig.add_trace(
                    go.Scatter(x=data.index, y=k_val, mode='lines', name=f"Stoch %K({stoch_k})"),
                    row=row_idx, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=d_val, mode='lines', line=dict(dash='dash'), name=f"Stoch %D({stoch_d})"),
                    row=row_idx, col=1
                )
            elif ind_name == "ATR":
                # Compute Average True Range to assess volatility
                atr_val = ta.volatility.AverageTrueRange(
                    high=data['High'], low=data['Low'], close=data['Close'], window=atr_length
                ).average_true_range()
                fig.add_trace(
                    go.Scatter(x=data.index, y=atr_val, mode='lines', name=f"ATR({atr_length})"),
                    row=row_idx, col=1
                )
            elif ind_name == "ADX":
                # Compute ADX and directional indicators to gauge trend strength
                adx_obj = ta.trend.ADXIndicator(
                    high=data['High'], low=data['Low'], close=data['Close'], window=adx_length
                )
                adx_val = adx_obj.adx()
                plus_di = adx_obj.adx_pos()
                minus_di= adx_obj.adx_neg()
                fig.add_trace(
                    go.Scatter(x=data.index, y=adx_val, mode='lines', name=f"ADX({adx_length})"),
                    row=row_idx, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=plus_di, mode='lines', line=dict(dash='dash'), name="+DI"),
                    row=row_idx, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=minus_di, mode='lines', line=dict(dash='dash'), name="-DI"),
                    row=row_idx, col=1
                )
            elif ind_name == "CCI":
                # Compute Commodity Channel Index
                cci_val = ta.trend.CCIIndicator(
                    high=data['High'], low=data['Low'], close=data['Close'], window=cci_length
                ).cci()
                fig.add_trace(
                    go.Scatter(x=data.index, y=cci_val, mode='lines', name=f"CCI({cci_length})"),
                    row=row_idx, col=1
                )
            elif ind_name == "Williams %R":
                # Compute Williams %R momentum indicator
                wr_val = ta.momentum.WilliamsRIndicator(
                    high=data['High'], low=data['Low'], close=data['Close'], lbp=wr_length
                ).williams_r()
                fig.add_trace(
                    go.Scatter(x=data.index, y=wr_val, mode='lines', name=f"Williams %R({wr_length})"),
                    row=row_idx, col=1
                )
            elif ind_name == "ROC":
                # Compute Rate of Change indicator
                roc_val = ta.momentum.ROCIndicator(close=data['Close'], window=roc_length).roc()
                fig.add_trace(
                    go.Scatter(x=data.index, y=roc_val, mode='lines', name=f"ROC({roc_length})"),
                    row=row_idx, col=1
                )
            elif ind_name == "MFI":
                # Compute Money Flow Index indicator
                mfi_val = ta.volume.MFIIndicator(
                    high=data['High'], low=data['Low'], close=data['Close'],
                    volume=data['Volume'], window=mfi_length
                ).money_flow_index()
                fig.add_trace(
                    go.Scatter(x=data.index, y=mfi_val, mode='lines', name=f"MFI({mfi_length})"),
                    row=row_idx, col=1
                )

        # Add overlay indicators to the main price chart
        for ov in [i for i in indicators if i in overlay_indicators_set]:
            add_overlay(ov)

        # Add oscillator indicators in their designated subplot rows and label y-axes
        osc_row_start = 3
        for i, osc_ind in enumerate(oscillator_list):
            row_idx = osc_row_start + i
            add_oscillator(osc_ind, row_idx)
            fig.update_yaxes(title_text=osc_ind, row=row_idx, col=1)

        # Update overall layout settings for aesthetics and legend placement
        fig.update_layout(
            template="plotly",
            height=1600,
            # title=f"{ticker} - {timeframe} Chart ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80)
        )
        fig.update_yaxes(title_text="Price + Overlays", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Store the generated figure in session state for later retrieval
        st.session_state[f"plotly_fig_{ticker}"] = fig
        return fig, None

    except Exception as e:
        # Handle exceptions by displaying an error message and returning an error analysis object
        st.error(f"General error in analyze_ticker: {e}")
        return (None, {
            "action": "Error",
            "confidence_score": 0,
            "price_target": "N/A",
            "justification": f"General Analysis Error: {e}"
        })

# ------------------------------------------------------------------------------
@st.cache_data(ttl=timedelta(minutes=30))
def fetch_stock_data(tickers, start_date, end_date, yf_interval):
    """
    Fetches historical stock data for each ticker using yfinance.
    Caches results for 30 minutes to reduce redundant API calls.

    Parameters:
        tickers (list): List of ticker symbols.
        start_date (date): Start date for data fetching.
        end_date (date): End date for data fetching.
        yf_interval (str): Interval string compatible with yfinance.

    Returns:
        dict: A dictionary with ticker symbols as keys and their corresponding data DataFrames.
    """
    adjusted_end = end_date
    if start_date == end_date:
        adjusted_end = start_date + timedelta(days=1)
    stock_data = {}
    with st.spinner("Fetching Stock Data..."):
        for t in tickers:
            data = yf.download(t, start=start_date, end=adjusted_end, interval=yf_interval,  multi_level_index=False)
            if not data.empty:
                stock_data[t] = data
            else:
                st.warning(f"No data found for {t}.")
    return stock_data

# ------------------------------------------------------------------------------
# Fetch Data
# ------------------------------------------------------------------------------
# Button trigger to fetch stock data and store it in session state
if st.sidebar.button("Fetch Data"):
    st.session_state["stock_data"] = fetch_stock_data(tickers, start_date, end_date, yf_interval)
    st.session_state.pop("analysis_results", None)  # Reset previous AI analysis results

# ------------------------------------------------------------------------------
# Generate & Display Charts
# ------------------------------------------------------------------------------
# Iterate over fetched stock data and generate Plotly charts if not already created
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    for tkr in st.session_state["stock_data"]:
        if f"plotly_fig_{tkr}" not in st.session_state:
            data_df = st.session_state["stock_data"][tkr]
            fig, _ = analyze_ticker(tkr, data_df, indicator_params, start_date, end_date)
            if fig:
                st.session_state[f"plotly_fig_{tkr}"] = fig

# ------------------------------------------------------------------------------
# Run AI Analysis
# ------------------------------------------------------------------------------
# Button trigger to perform AI-based technical analysis using the generated charts
if st.sidebar.button("Run AI Analysis"):
    if "stock_data" not in st.session_state or not st.session_state["stock_data"]:
        st.warning("Please fetch stock data first.")
    else:
        analysis_results = {}
        with st.spinner("Running AI Technical Analysis..."):
            for tkr in st.session_state["stock_data"]:
                data_df = st.session_state["stock_data"][tkr]
                fig_result = analyze_ticker(tkr, data_df, indicator_params, start_date, end_date)
                if fig_result:
                    fig, _ = fig_result

                    # Write the Plotly figure as a PNG image to a temporary file using kaleido
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        fig.write_image(tmpfile.name, format="png", engine="kaleido")
                        tmp_path = tmpfile.name
                    with open(tmp_path, "rb") as f:
                        img_bytes = f.read()
                    os.remove(tmp_path)

                    with st.spinner("AI Analyzing Chart..."):
                        # Use our model.py function to analyze the chart
                        try:
                            result = generate_technical_analysis(img_bytes, tkr, timeframe, selected_indicators_code)
                            analysis_results[tkr] = result
                        except Exception as e:
                            st.error(f"Error during AI analysis: {str(e)}")
                            analysis_results[tkr] = {
                                "action": "Error",
                                "confidence_score": 0,
                                "price_target": "N/A",
                                "justification": f"AI Analysis Error: {str(e)}"
                            }

                else:
                    analysis_results[tkr] = {
                        "action":"Error","confidence_score":0,"price_target":"N/A",
                        "justification":"AI Analysis Failed. See error above."
                    }
        st.session_state["analysis_results"] = analysis_results
        st.success("AI Technical Analysis Completed!")

# ------------------------------------------------------------------------------
# Final Tabs
# ------------------------------------------------------------------------------
# Display the overall summary and individual stock analysis in tabbed format
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    tickers_list = list(st.session_state["stock_data"].keys())
    tab_names = ["Overall Summary"] + tickers_list
    tabs = st.tabs(tab_names)
    overall_results = []
    analysis_results = st.session_state.get("analysis_results", {})

    # Loop through each ticker to render its analysis and chart
    for i, tkr in enumerate(tickers_list):
        result = analysis_results.get(tkr, {})
        fig = st.session_state.get(f"plotly_fig_{tkr}")
        if fig is not None:
            overall_results.append({
                "Stock": tkr,
                "Recommendation": result.get("action", "N/A"),
                "Confidence": result.get("confidence_score", "N/A")
            })
            with tabs[i + 1]:
                st.subheader(f"Analysis for {tkr} ({timeframe})")
                st.plotly_chart(fig)

                if tkr in analysis_results:
                    st.write("**AI Recommendation:**", result.get("action", "N/A"))
                    st.write("**Confidence Score (1-10):**", result.get("confidence_score", "N/A"))
                    st.write("**Detailed Justification:**")
                    st.write(result.get("justification", "No justification provided."))
                else:
                    st.info("AI analysis not yet run.")

                # -------------- Prevent app rerun by using a base64 link instead of st.download_button -------------
                buf = BytesIO()
                fig.write_image(buf, format="png", engine="kaleido")
                buf.seek(0)

                # Encode the image bytes as a base64 string for download
                b64_data = base64.b64encode(buf.read()).decode("utf-8")

                download_filename = f"{tkr}_chart.png"
                download_link = f'<a href="data:image/png;base64,{b64_data}" download="{download_filename}">**Download Chart as PNG**</a>'

                st.markdown(download_link, unsafe_allow_html=True)

    # Display overall summary tab with a table of recommendations
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        if "analysis_results" not in st.session_state:
            st.info("Click 'Run AI Analysis' to see results")
        elif overall_results:
            df_summary = pd.DataFrame(overall_results)
            st.table(df_summary[['Stock', 'Recommendation', 'Confidence']])
        else:
            st.info("No stocks selected or analyzed yet.")
else:
    st.info("Please fetch stock data and then run AI Analysis to see results.")
