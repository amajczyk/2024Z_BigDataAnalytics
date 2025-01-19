import streamlit as st
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.util import Date
import plotly.express as px
import datetime
import plotly.graph_objects as go
import json
from datetime import timedelta
import numpy as np
import plotly.express as px
from datetime import timedelta


# Connect to Cassandra
@st.cache_resource
def get_cassandra_session():
    cluster = Cluster(["127.0.0.1"])
    session = cluster.connect()
    session.set_keyspace("gold_layer")
    return session

@st.cache_resource
def get_cassandra_session_stream():
    cluster = Cluster(["127.0.0.1"])
    session = cluster.connect()
    session.set_keyspace("stream_predictions")
    return session

# Load Batch Data from Cassandra
# @st.cache_data
def load_batch_data():
    print("Loading batch data from Cassandra")
    session = get_cassandra_session()

    news_query = "SELECT * FROM aggregated_news"
    news_rows = session.execute(news_query)
    news_df = pd.DataFrame(list(news_rows))

    yfinance_query = "SELECT * FROM aggregated_yfinance"
    yfinance_rows = session.execute(yfinance_query)
    yfinance_df = pd.DataFrame(list(yfinance_rows))

    keywords_query = "SELECT * FROM aggregated_keywords"
    keywords_rows = session.execute(keywords_query)
    keywords_df = pd.DataFrame(list(keywords_rows))

    # Convert Cassandra Date type to datetime.date
    for df in [news_df, yfinance_df, keywords_df]:
        if "aggregation_date" in df.columns:
            df["aggregation_date"] = df["aggregation_date"].apply(lambda x: x if isinstance(x, datetime.date) else x.date())

    return news_df, yfinance_df, keywords_df

# Load Stream Data from Cassandra
# @st.cache_data
def load_stream_data():
    print("Loading model predictions from Cassandra...")
    session = get_cassandra_session_stream()
    
    query = "SELECT * FROM model_predictions_10m"
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"])

    df['actual_price'] = df['input_data'].apply(lambda x: json.loads(x)['price'])
    return df

# Streamlit App
st.set_page_config(page_title="Data Aggregation Dashboard", layout="wide")
st.title("Data Aggregation Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Batch", "Stream"])


# Batch Tab
with tab1:
    
    st.header("Batch Analysis")
    
    # Load batch data only when in Batch tab
    news_df, yfinance_df, keywords_df = load_batch_data()

    # Filter by aggregation_date
    min_date = max(
        news_df["aggregation_date"].min(),
        yfinance_df["aggregation_date"].min(),
        keywords_df["aggregation_date"].min()
    )
    max_date = min(
        news_df["aggregation_date"].max(),
        yfinance_df["aggregation_date"].max(),
        keywords_df["aggregation_date"].max()
    )

    selected_date_range = st.slider(
        "Select Aggregation Date Range:", 
        min_value=min_date, 
        max_value=max_date, 
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter dataframes by selected date range
    start_date, end_date = selected_date_range
    news_df = news_df[(news_df["aggregation_date"] >= start_date) & (news_df["aggregation_date"] <= end_date)]
    yfinance_df = yfinance_df[(yfinance_df["aggregation_date"] >= start_date) & (yfinance_df["aggregation_date"] <= end_date)]
    keywords_df = keywords_df[(keywords_df["aggregation_date"] >= start_date) & (keywords_df["aggregation_date"] <= end_date)]

    # KPIs
    total_articles = news_df["total_articles"].sum()
    total_companies = yfinance_df["symbol"].nunique()
    avg_stock_price = yfinance_df["avg_stock_price"].mean()
    top_keyword = keywords_df.groupby("keyword")["count"].sum().idxmax() if not keywords_df.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", f"{total_articles}")
    col2.metric("Total Companies", f"{total_companies}")
    col3.metric("Avg. Stock Price", f"{avg_stock_price:.2f}")
    col4.metric("Top Keyword", top_keyword)

    # Plots
    st.subheader("Insights")

    # 1. Articles per Source
    articles_per_source = news_df.groupby("symbol")["total_articles"].sum().reset_index()
    fig1 = px.bar(articles_per_source, x="symbol", y="total_articles", title="Articles per Source Site")
    st.plotly_chart(fig1)

    # 2. Stock Price Trends
    stock_trends = yfinance_df.groupby(["symbol", "aggregation_date"])["avg_stock_price"].mean().reset_index()
    fig2 = px.line(stock_trends, x="aggregation_date", y="avg_stock_price", color="symbol", title="Stock Price Trends")
    st.plotly_chart(fig2)

    # 3. Volume Traded per Company
    volume_per_company = yfinance_df.groupby("symbol")["volume_traded"].sum().reset_index()
    fig3 = px.bar(volume_per_company, x="symbol", y="volume_traded", title="Volume Traded per Company")
    st.plotly_chart(fig3)

    # 4. Keyword Counts
    keyword_counts = keywords_df.groupby("keyword")["count"].sum().reset_index().sort_values(by="count", ascending=False).head(10)
    fig4 = px.bar(keyword_counts, x="keyword", y="count", title="Top 10 Keywords")
    st.plotly_chart(fig4)

    # 5. Articles Over Time
    articles_over_time = news_df.groupby("aggregation_date")["total_articles"].sum().reset_index()
    fig5 = px.line(articles_over_time, x="aggregation_date", y="total_articles", title="Articles Over Time")
    st.plotly_chart(fig5)

    # 6. Average Volatility by Company
    avg_volatility = yfinance_df.groupby("symbol")["avg_volatility"].mean().reset_index()
    fig6 = px.bar(avg_volatility, x="symbol", y="avg_volatility", title="Average Volatility by Company")
    st.plotly_chart(fig6)

    # 7. Stock Price Distribution
    fig7 = px.box(yfinance_df, x="symbol", y="avg_stock_price", title="Stock Price Distribution by Company")
    st.plotly_chart(fig7)

    print(keywords_df.head())


    # 8. Treemap for Keyword Counts
    keywords_filtered = keywords_df[keywords_df["count"] > 1]
    fig8 = px.treemap(
        keywords_filtered,
        path=["keyword"],
        values="count",
        title="Keyword Distribution Treemap",
        hover_data={"count": True, "keyword": True},
    )
    fig8.update_traces(
        texttemplate="%{label}"  # Display keyword name only if count > 1
    )
    st.plotly_chart(fig8)





with tab2:
    st.header("Real-time Market Analysis")
    
    predictions_df = load_stream_data()
    available_symbols = predictions_df["symbol"].unique().tolist()
    selected_symbol = st.selectbox("Select Symbol", available_symbols)
    
    # Calculate latest model RMSE for selected symbol
    symbol_predictions = predictions_df[predictions_df["symbol"] == selected_symbol]
    valid_predictions = symbol_predictions[symbol_predictions["label"].notna()]
    model_rmse = np.sqrt(((valid_predictions["prediction"] - valid_predictions["label"]) ** 2).mean())
    
    # Display model RMSE for selected symbol
    st.markdown(f'<div style="padding:10px;border-radius:5px;background:#FFB6C1;width:250px;margin-bottom:20px">',unsafe_allow_html=True)
    st.metric(f"Latest Model RMSE for {selected_symbol}", f"{model_rmse:.4f}")
    st.markdown('</div>',unsafe_allow_html=True)
    
    symbol_count = len(predictions_df[predictions_df["symbol"] == selected_symbol])
    st.markdown(f'<div style="padding:10px;border-radius:5px;background:#4ECDC4;width:200px;text-align:center">',unsafe_allow_html=True)
    st.metric(f"{selected_symbol}", f"{symbol_count} records")
    st.markdown('</div>',unsafe_allow_html=True)
    
    filtered_df = predictions_df[predictions_df["symbol"] == selected_symbol].copy()
    filtered_df.sort_values(by="event_time", inplace=True)
    filtered_df["input_data"] = filtered_df["input_data"].apply(json.loads)
    metrics_df = pd.json_normalize(filtered_df["input_data"])

    if selected_symbol == "ETHEREUM":
        # Create actual price and label dataframes with 10-minute offset
        actual_df = filtered_df[filtered_df["event_time"] - timedelta(minutes=10) >= filtered_df["event_time"].min()].copy()
        actual_df["event_time"] = actual_df["event_time"] - timedelta(minutes=10)
        
        label_df = filtered_df[filtered_df["label"].notna()].copy()
        label_df["event_time"] = label_df["event_time"] - timedelta(minutes=10)
        
        actual_prices = metrics_df["price"].tolist()
        filtered_df["actual_price"] = actual_prices
        filtered_df = filtered_df[filtered_df["prediction"] != -1]
        
        # Price prediction chart
        pred_plot = go.Figure()
        pred_plot.add_trace(go.Scatter(x=filtered_df["event_time"], y=filtered_df["prediction"], 
                                    name="Predicted", line=dict(color="#FF6B6B")))
        pred_plot.add_trace(go.Scatter(x=label_df["event_time"], y=label_df["label"], 
                                    name="Expected", line=dict(color="#4ECDC4")))
        pred_plot.add_trace(go.Scatter(x=actual_df["event_time"], y=actual_df["actual_price"], 
                                    name="Actual", line=dict(color="#45B7D1")))

        pred_plot.update_layout(
            title=f"Price Prediction vs Actual for {selected_symbol}",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        st.plotly_chart(pred_plot)

        # Ethereum specific charts
        col1, col2 = st.columns(2)
        with col1:
            fig_spread = px.line(metrics_df, y="spread_raw", 
                                title="Spread Raw",
                                line_shape="spline")
            fig_spread.update_traces(line_color="#FF6B6B", line_width=2)
            st.plotly_chart(fig_spread)
            
        with col2:
            fig_bid_ask = go.Figure()
            fig_bid_ask.add_trace(go.Scatter(y=metrics_df["bid"], name="Bid", line=dict(color="#4ECDC4")))
            fig_bid_ask.add_trace(go.Scatter(y=metrics_df["ask"], name="Ask", line=dict(color="#45B7D1")))
            fig_bid_ask.update_layout(title="Bid/Ask Prices")
            st.plotly_chart(fig_bid_ask)

        # Correlation matrices for Ethereum
        corr_matrix = metrics_df[["bid", "ask", "spread_raw", "spread_table", "price"]].corr().round(3)
        spearman_corr = metrics_df[["bid", "ask", "spread_raw", "spread_table", "price"]].corr(method='spearman').round(3)

    else:
        # Create actual price and label dataframes with 10-minute offset
        actual_df = filtered_df[filtered_df["event_time"] - timedelta(minutes=10) >= filtered_df["event_time"].min()].copy()
        actual_df["event_time"] = actual_df["event_time"] - timedelta(minutes=10)
        
        label_df = filtered_df[filtered_df["label"].notna()].copy()
        label_df["event_time"] = label_df["event_time"] - timedelta(minutes=10)
        
        actual_prices = metrics_df["price"].tolist()
        filtered_df["actual_price"] = actual_prices
        
        # Price prediction chart for other symbols
        pred_plot = px.line(filtered_df, x="event_time", y="prediction", 
                           title=f"Price Prediction vs Actual for {selected_symbol}")
        label_plot = px.line(label_df, x="event_time", y="label")
        actual_plot = px.line(actual_df, x="event_time", y="actual_price")
        
        pred_plot.add_traces(label_plot.data)
        pred_plot.add_traces(actual_plot.data)
        
        pred_plot.data[0].line.color = "#FF6B6B"
        pred_plot.data[1].line.color = "#4ECDC4"  
        pred_plot.data[2].line.color = "#45B7D1"
        
        pred_plot.data[0].name = "Predicted"
        pred_plot.data[1].name = "Expected"
        pred_plot.data[2].name = "Actual"
        
        pred_plot.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        st.plotly_chart(pred_plot)

        # Other symbols charts
        col1, col2 = st.columns(2)
        with col1:
            fig_sentiment = px.line(metrics_df, y="market_sentiment", 
                                    title="Market Sentiment Trend",
                                    line_shape="spline")
            fig_sentiment.update_traces(line_color="#FF6B6B", line_width=2)
            st.plotly_chart(fig_sentiment)
            
        with col2:
            fig_volume = px.bar(metrics_df, y="volume", 
                                title="Trading Volume",
                                color_discrete_sequence=["#4ECDC4"])
            st.plotly_chart(fig_volume)

        # Correlation matrices for other symbols
        corr_matrix = metrics_df[["price", "volume", "volatility", "market_sentiment", "trading_activity"]].corr().round(3)
        spearman_corr = metrics_df[["price", "volume", "volatility", "market_sentiment", "trading_activity"]].corr(method='spearman').round(3)

        # Additional charts for other symbols
        col3, col4 = st.columns(2)
        with col3:
            fig_activity = px.line(metrics_df, y="trading_activity", 
                                title="Trading Activity",
                                line_shape="spline")
            fig_activity.update_traces(line_color="#96CEB4", line_width=2)
            st.plotly_chart(fig_activity)
            
        with col4:
            fig_volatility = px.line(metrics_df, y="volatility", 
                                    title="Volatility",
                                    line_shape="spline")
            fig_volatility.update_traces(line_color="#FFEEAD", line_width=2)
            st.plotly_chart(fig_volatility)

    # Display correlation matrices
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linear Correlations")
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdYlBu",
            text=corr_matrix.values,
            texttemplate="%{text:.3f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig_corr.update_layout(height=400, width=400)
        st.plotly_chart(fig_corr)

    with col2:
        st.subheader("Spearman Rank Correlations")
        fig_spearman = go.Figure(data=go.Heatmap(
            z=spearman_corr.values,
            x=spearman_corr.columns,
            y=spearman_corr.columns,
            colorscale="Viridis",
            text=spearman_corr.values,
            texttemplate="%{text:.3f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig_spearman.update_layout(height=400, width=400)
        st.plotly_chart(fig_spearman)