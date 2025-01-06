import streamlit as st
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.util import Date
import plotly.express as px
import datetime

# Connect to Cassandra
@st.cache_resource
def get_cassandra_session():
    cluster = Cluster(["127.0.0.1"])
    session = cluster.connect()
    session.set_keyspace("gold_layer")
    return session

def get_cassandra_session_stream():
    cluster = Cluster(["127.0.0.1"])
    session = cluster.connect()
    session.set_keyspace("stream_predictions")
    return session

# Load Data from Cassandra
# @st.cache_data
def load_data():
    print("Loading data from Cassandra")
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


def load_predictions():
    """Fetch model predictions from Cassandra and return a DataFrame."""
    print("Loading model predictions from Cassandra...")
    session = get_cassandra_session_stream()
    
    query = "SELECT * FROM model_predictions"
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))

    # Ensure event_time is converted to a proper datetime
    # (Depending on how Cassandra returns the timestamp, this might be optional)
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"])

    return df



# Streamlit App
st.set_page_config(page_title="Data Aggregation Dashboard", layout="wide")
st.title("Data Aggregation Dashboard")

# Tabs
tab1, tab2 = st.tabs(["Batch", "Stream"])

# Batch Tab
with tab1:
    st.header("Batch Analysis")

    # Load Data
    news_df, yfinance_df, keywords_df = load_data()

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

    # # 8. Keyword Distribution Over Time with Animation
    # keywords_over_time = keywords_df.groupby(["aggregation_date", "keyword"])["count"].sum().reset_index()
    # fig8 = px.scatter(
    #     keywords_over_time, 
    #     x="aggregation_date", 
    #     y="count", 
    #     color="keyword", 
    #     size="count",
    #     animation_frame="aggregation_date",
    #     title="Keyword Distribution Over Time (Animated)",
    #     range_x=[min_date, max_date],  # Set constant x-axis limits
    #     range_y=[0, keywords_over_time["count"].max()],  # Set dynamic y-axis limits
    #     opacity=0.5
    # )
    # fig8.update_layout(xaxis_title="Days", yaxis_title="Keyword Counts")
    # st.plotly_chart(fig8)

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

# Stream Tab
with tab2:
   
    st.header("Expected vs. Predicted Price")

    # Load model predictions DataFrame
    predictions_df = load_predictions()

    # Allow user to pick which symbol to visualize
    available_symbols = predictions_df["symbol"].unique().tolist()
    selected_symbol = st.selectbox("Select Symbol", available_symbols)

    # Filter & sort
    filtered_df = predictions_df[predictions_df["symbol"] == selected_symbol].copy()
    filtered_df.sort_values(by="event_time", inplace=True)

    # Plot with Plotly Express
    # Option 1: Passing multiple y-values
    fig = px.line(
        filtered_df, 
        x="event_time", 
        y=["label", "prediction"], 
        title=f"Expected vs. Predicted Price for {selected_symbol}"
    )
    st.plotly_chart(fig)

    # Option 2: Melt the DataFrame for a single line
    # df_long = filtered_df.melt(
    #     id_vars="event_time", 
    #     value_vars=["label", "prediction"], 
    #     var_name="Type", 
    #     value_name="Price"
    # )
    # fig = px.line(df_long, x="event_time", y="Price", color="Type", 
    #               title=f"Expected vs. Predicted Price for {selected_symbol}")
    # st.plotly_chart(fig)
