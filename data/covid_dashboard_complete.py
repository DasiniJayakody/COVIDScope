#!/usr/bin/env python3
"""
COVID-19 Complete Interactive Dashboard
Comprehensive Streamlit web application with all features
Combines single country analysis, multi-country comparison, and global overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="COVID-19 Complete Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load and cache the COVID-19 dataset"""
    try:
        df = pd.read_csv("owid-covid-data.csv")
        df.columns = ["country", "date", "new_cases_per_million"]
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def get_available_countries(df):
    """Get list of countries with recent COVID activity and major countries"""
    if df is None:
        return []

    # Major countries list
    major_countries = [
        "United States",
        "China",
        "India",
        "Japan",
        "Germany",
        "United Kingdom",
        "France",
        "Italy",
        "Canada",
        "Brazil",
        "Russia",
        "South Korea",
        "Australia",
        "Spain",
        "Mexico",
        "Indonesia",
        "Netherlands",
        "Saudi Arabia",
        "Turkey",
        "Switzerland",
        "Argentina",
        "Sweden",
        "Belgium",
        "Thailand",
        "Israel",
        "Austria",
        "Norway",
        "Singapore",
        "Malaysia",
        "Denmark",
        "South Africa",
        "Philippines",
        "Egypt",
        "Vietnam",
        "Pakistan",
        "Iran",
        "Colombia",
        "Chile",
        "Poland",
        "Ukraine",
        "Romania",
        "Czech Republic",
        "Portugal",
        "Greece",
        "Hungary",
        "Bulgaria",
        "Croatia",
        "Slovakia",
        "Slovenia",
        "Lithuania",
        "Latvia",
        "Estonia",
        "Finland",
        "Iceland",
        "Luxembourg",
        "Malta",
        "Cyprus",
        "Sri Lanka",
    ]

    # Get all unique countries from dataset
    all_countries = df["country"].unique().tolist()

    # Get countries with recent activity
    recent_data = df[df["date"] >= "2024-01-01"]
    country_activity = (
        recent_data.groupby("country")["new_cases_per_million"]
        .agg(["mean", "max"])
        .sort_values("max", ascending=False)
    )
    active_countries = country_activity[country_activity["max"] > 10].head(50)

    # Combine major countries with active countries
    combined_countries = list(set(major_countries + active_countries.index.tolist()))

    # Filter to only include countries that exist in the dataset
    available_countries = [
        country for country in combined_countries if country in all_countries
    ]

    # Sort alphabetically
    available_countries.sort()

    return available_countries


def calculate_statistics(country_df):
    """Calculate comprehensive statistics for a country"""
    if country_df.empty:
        return {}

    # Basic statistics
    total_cases_per_million = country_df["new_cases_per_million"].sum()
    mean_daily_cases = country_df["new_cases_per_million"].mean()
    max_daily_cases = country_df["new_cases_per_million"].max()
    std_daily_cases = country_df["new_cases_per_million"].std()

    # Recent trends (last 30 days vs previous 30 days)
    recent_30d = country_df.tail(30)["new_cases_per_million"].mean()
    previous_30d = (
        country_df.iloc[-60:-30]["new_cases_per_million"].mean()
        if len(country_df) >= 60
        else recent_30d
    )

    trend_change = (
        ((recent_30d - previous_30d) / previous_30d * 100) if previous_30d > 0 else 0
    )

    # 7-day moving average
    recent_7d_avg = country_df.tail(7)["new_cases_per_million"].mean()

    # Growth rate (last 7 days vs previous 7 days)
    last_7d = country_df.tail(7)["new_cases_per_million"].sum()
    prev_7d = (
        country_df.iloc[-14:-7]["new_cases_per_million"].sum()
        if len(country_df) >= 14
        else last_7d
    )
    growth_rate = ((last_7d - prev_7d) / prev_7d * 100) if prev_7d > 0 else 0

    return {
        "total_cases_per_million": total_cases_per_million,
        "mean_daily_cases": mean_daily_cases,
        "max_daily_cases": max_daily_cases,
        "std_daily_cases": std_daily_cases,
        "trend_change": trend_change,
        "recent_7d_avg": recent_7d_avg,
        "growth_rate": growth_rate,
        "last_7d": last_7d,
        "prev_7d": prev_7d,
    }


def create_comparison_chart(countries_data, countries_list):
    """Create comparison chart for multiple countries"""
    fig = go.Figure()

    colors = px.colors.qualitative.Set3
    for i, country in enumerate(countries_list):
        if country in countries_data and not countries_data[country].empty:
            data = countries_data[country]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["new_cases_per_million"],
                    mode="lines",
                    name=country,
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f"<b>{country}</b><br>Date: %{{x}}<br>Cases per million: %{{y:.1f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"COVID-19 Cases Comparison: {', '.join(countries_list)}",
        xaxis_title="Date",
        yaxis_title="New Cases per Million",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_heatmap_chart(countries_data, countries_list, start_date, end_date):
    """Create heatmap for multiple countries"""
    # Prepare data for heatmap
    heatmap_data = []
    dates = []

    for country in countries_list:
        if country in countries_data and not countries_data[country].empty:
            data = countries_data[country]
            # Filter by date range
            mask = (data.index >= start_date) & (data.index <= end_date)
            filtered_data = data[mask]

            if not filtered_data.empty:
                # Resample to weekly data for better visualization
                weekly_data = filtered_data.resample("W").mean()
                heatmap_data.append(weekly_data["new_cases_per_million"].values)
                if not dates:
                    dates = [d.strftime("%Y-%m-%d") for d in weekly_data.index]

    if heatmap_data and dates:
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=dates,
                y=countries_list,
                colorscale="Reds",
                hovertemplate="<b>%{y}</b><br>Week: %{x}<br>Cases per million: %{z:.1f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="COVID-19 Cases Heatmap (Weekly Average)",
            xaxis_title="Week",
            yaxis_title="Country",
            height=400,
        )

        return fig

    return None


def create_ranking_chart(countries_data, countries_list, metric="total_cases"):
    """Create ranking chart for countries"""
    rankings = []

    for country in countries_list:
        if country in countries_data and not countries_data[country].empty:
            data = countries_data[country]
            if metric == "total_cases":
                value = data["new_cases_per_million"].sum()
            elif metric == "max_cases":
                value = data["new_cases_per_million"].max()
            elif metric == "avg_cases":
                value = data["new_cases_per_million"].mean()
            elif metric == "recent_activity":
                value = data.tail(30)["new_cases_per_million"].mean()

            rankings.append({"country": country, "value": value})

    if rankings:
        rankings.sort(key=lambda x: x["value"], reverse=True)
        countries = [r["country"] for r in rankings]
        values = [r["value"] for r in rankings]

        fig = go.Figure(
            data=go.Bar(
                x=values,
                y=countries,
                orientation="h",
                marker_color="lightcoral",
                hovertemplate="<b>%{y}</b><br>Value: %{x:.1f}<extra></extra>",
            )
        )

        metric_names = {
            "total_cases": "Total Cases per Million",
            "max_cases": "Maximum Daily Cases",
            "avg_cases": "Average Daily Cases",
            "recent_activity": "Recent Activity (30-day avg)",
        }

        fig.update_layout(
            title=f"Country Rankings by {metric_names.get(metric, metric)}",
            xaxis_title=metric_names.get(metric, metric),
            yaxis_title="Country",
            height=400,
        )

        return fig

    return None


def create_daily_cases_chart(country_df, country_name):
    """Create daily cases chart"""
    fig = go.Figure()

    # Daily cases
    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=country_df["new_cases_per_million"],
            mode="lines",
            name="Daily Cases",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Date: %{x}<br>Cases per million: %{y:.1f}<extra></extra>",
        )
    )

    # 7-day moving average
    ma7 = country_df["new_cases_per_million"].rolling(window=7).mean()
    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=ma7,
            mode="lines",
            name="7-Day Moving Average",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate="Date: %{x}<br>7-day avg: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Daily COVID-19 Cases in {country_name}",
        xaxis_title="Date",
        yaxis_title="New Cases per Million",
        hovermode="x unified",
        height=400,
    )

    return fig


def create_cumulative_chart(country_df, country_name):
    """Create cumulative cases chart"""
    cumulative_cases = country_df["new_cases_per_million"].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=cumulative_cases,
            mode="lines",
            name="Cumulative Cases",
            line=dict(color="#2ca02c", width=3),
            fill="tonexty",
            hovertemplate="Date: %{x}<br>Cumulative cases per million: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Cumulative COVID-19 Cases in {country_name}",
        xaxis_title="Date",
        yaxis_title="Cumulative Cases per Million",
        height=400,
    )

    return fig


def create_weekly_summary_chart(country_df, country_name):
    """Create weekly summary chart"""
    weekly_data = (
        country_df.resample("W")
        .agg({"new_cases_per_million": ["sum", "mean", "max"]})
        .round(2)
    )

    weekly_data.columns = ["Weekly Total", "Weekly Average", "Weekly Peak"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=weekly_data.index,
            y=weekly_data["Weekly Total"],
            name="Weekly Total",
            marker_color="lightblue",
            hovertemplate="Week: %{x}<br>Total: %{y:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=weekly_data.index,
            y=weekly_data["Weekly Peak"],
            mode="lines+markers",
            name="Weekly Peak",
            line=dict(color="red"),
            hovertemplate="Week: %{x}<br>Peak: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Weekly COVID-19 Summary in {country_name}",
        xaxis_title="Week",
        yaxis_title="Cases per Million",
        barmode="group",
        height=400,
    )

    return fig


def create_trend_analysis_chart(country_df, country_name):
    """Create trend analysis chart"""
    # Calculate moving averages
    ma7 = country_df["new_cases_per_million"].rolling(window=7).mean()
    ma30 = country_df["new_cases_per_million"].rolling(window=30).mean()

    # Calculate growth rate
    growth_rate = country_df["new_cases_per_million"].pct_change() * 100
    growth_rate = growth_rate.replace([np.inf, -np.inf], np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Moving Averages", "Growth Rate (%)"),
        vertical_spacing=0.1,
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=country_df["new_cases_per_million"],
            mode="lines",
            name="Daily Cases",
            line=dict(color="lightgray", width=1),
            opacity=0.5,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=ma7,
            mode="lines",
            name="7-Day MA",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=ma30,
            mode="lines",
            name="30-Day MA",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    # Growth rate
    fig.add_trace(
        go.Scatter(
            x=country_df.index,
            y=growth_rate,
            mode="lines",
            name="Growth Rate",
            line=dict(color="green", width=2),
            fill="tonexty",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=f"Trend Analysis for {country_name}", height=600, showlegend=True
    )

    return fig


def create_prediction_chart(country_df, country_name):
    """Create linear regression prediction chart"""
    if len(country_df) < 14:
        return None

    # Use last 14 days for prediction
    recent_data = country_df.tail(14)
    x = np.arange(len(recent_data))
    y = recent_data["new_cases_per_million"].values

    # Simple linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator != 0:
        m = numerator / denominator
        c = y_mean - m * x_mean

        # Predict next 7 days
        future_x = np.arange(len(x), len(x) + 7)
        future_y = m * future_x + c

        # Calculate R-squared
        y_pred = m * x + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        fig = go.Figure()

        # Actual data
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name="Actual Data",
                line=dict(color="blue", width=3),
                hovertemplate="Day: %{x}<br>Cases: %{y:.1f}<extra></extra>",
            )
        )

        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=future_y,
                mode="lines+markers",
                name="Prediction (Next 7 Days)",
                line=dict(color="red", width=3, dash="dash"),
                hovertemplate="Day: %{x}<br>Predicted: %{y:.1f}<extra></extra>",
            )
        )

        # Extend actual trend line
        all_x = np.concatenate([x, future_x])
        all_y_pred = m * all_x + c
        fig.add_trace(
            go.Scatter(
                x=all_x,
                y=all_y_pred,
                mode="lines",
                name="Trend Line",
                line=dict(color="green", width=2),
                opacity=0.7,
            )
        )

        fig.update_layout(
            title=f"COVID-19 Prediction for {country_name} (R² = {r_squared:.3f})",
            xaxis_title="Days",
            yaxis_title="Cases per Million",
            height=400,
        )

        return fig

    return None


def export_chart_as_png(fig):
    """Export chart as PNG"""
    img_bytes = fig.to_image(format="png")
    return img_bytes


def export_data_as_csv(data, filename):
    """Export data as CSV"""
    csv = data.to_csv(index=True)
    return csv


def main():
    """Main application"""
    st.markdown(
        '<h1 class="main-header">🦠 COVID-19 Complete Analysis Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading COVID-19 data..."):
        df = load_data()

    if df is None:
        st.error("Failed to load data. Please check if the data file exists.")
        return

    # Sidebar
    st.sidebar.markdown("## 📊 Dashboard Controls")

    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "🔍 Analysis Mode",
        options=["Single Country Analysis", "Country Comparison", "Global Overview"],
        index=0,
        help="Choose your analysis type",
    )

    # Date range selection
    st.sidebar.markdown("### 📅 Date Range")
    date_range = st.sidebar.selectbox(
        "Select Analysis Period",
        options=[
            "All Time",
            "Last 6 Months",
            "Last 3 Months",
            "Last Month",
            "Custom Range",
        ],
        index=0,
    )

    # Custom date range
    if date_range == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=df["date"].min().date())
        with col2:
            end_date = st.date_input("End Date", value=df["date"].max().date())
    else:
        end_date = df["date"].max().date()
        if date_range == "Last Month":
            start_date = end_date - timedelta(days=30)
        elif date_range == "Last 3 Months":
            start_date = end_date - timedelta(days=90)
        elif date_range == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        else:  # All Time
            start_date = df["date"].min().date()

    # Convert to datetime for filtering
    start_date_dt = pd.Timestamp(start_date)
    end_date_dt = pd.Timestamp(end_date)

    if analysis_mode == "Single Country Analysis":
        # Single country analysis
        available_countries = get_available_countries(df)

        # Add search functionality
        st.sidebar.markdown("### 🔍 Quick Search")
        search_term = st.sidebar.text_input(
            "Search for a country:", placeholder="Type country name..."
        )

        if search_term:
            filtered_countries = [
                country
                for country in available_countries
                if search_term.lower() in country.lower()
            ]
        else:
            filtered_countries = available_countries

        selected_country = st.sidebar.selectbox(
            "🌍 Select Country",
            options=filtered_countries,
            index=filtered_countries.index("United States")
            if "United States" in filtered_countries
            else 0,
        )

        # Show total countries available
        st.sidebar.markdown(
            f"**📊 Total Countries Available:** {len(available_countries)}"
        )

        # Filter data
        country_df = df[df["country"] == selected_country].copy()
        country_df = country_df[
            (country_df["date"] >= start_date_dt) & (country_df["date"] <= end_date_dt)
        ]
        country_df.set_index("date", inplace=True)

        # Calculate statistics
        stats = calculate_statistics(country_df)

        if country_df.empty:
            st.warning(
                f"No data available for {selected_country} in the selected date range."
            )
            return

        # Key metrics
        st.markdown("## 📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Cases per Million",
                value=f"{stats['total_cases_per_million']:,.0f}",
                delta=f"{stats['trend_change']:+.1f}%",
            )

        with col2:
            st.metric(
                label="Average Daily Cases",
                value=f"{stats['mean_daily_cases']:.1f}",
                delta=f"{stats['recent_7d_avg']:.1f} (7-day avg)",
            )

        with col3:
            st.metric(
                label="Peak Daily Cases",
                value=f"{stats['max_daily_cases']:.1f}",
                delta=f"{stats['std_daily_cases']:.1f} (std dev)",
            )

        with col4:
            st.metric(
                label="Growth Rate (7-day)",
                value=f"{stats['growth_rate']:+.1f}%",
                delta=f"{stats['last_7d']:.1f} vs {stats['prev_7d']:.1f}",
            )

        # Charts
        st.markdown("## 📊 Visualizations")

        # Daily cases chart
        st.markdown("### 📈 Daily Cases Trend")
        daily_chart = create_daily_cases_chart(country_df, selected_country)
        st.plotly_chart(daily_chart, use_container_width=True)

        # Export daily chart
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Daily Chart as PNG"):
                img_bytes = export_chart_as_png(daily_chart)
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name=f"{selected_country}_daily_cases.png",
                    mime="image/png",
                )

        with col2:
            if st.button("📥 Export Data as CSV"):
                csv_data = export_data_as_csv(
                    country_df, f"{selected_country}_data.csv"
                )
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{selected_country}_data.csv",
                    mime="text/csv",
                )

        # Cumulative chart
        st.markdown("### 📊 Cumulative Cases")
        cumulative_chart = create_cumulative_chart(country_df, selected_country)
        st.plotly_chart(cumulative_chart, use_container_width=True)

        # Weekly summary
        st.markdown("### 📅 Weekly Summary")
        weekly_chart = create_weekly_summary_chart(country_df, selected_country)
        st.plotly_chart(weekly_chart, use_container_width=True)

        # Trend analysis
        st.markdown("### 🔍 Trend Analysis")
        trend_chart = create_trend_analysis_chart(country_df, selected_country)
        st.plotly_chart(trend_chart, use_container_width=True)

        # Prediction
        st.markdown("### 🔮 Future Prediction")
        prediction_chart = create_prediction_chart(country_df, selected_country)
        if prediction_chart:
            st.plotly_chart(prediction_chart, use_container_width=True)
            st.info(
                "💡 Prediction uses simple linear regression on the last 14 days of data."
            )
        else:
            st.warning(
                "Not enough data for prediction (need at least 14 days of data)."
            )

    elif analysis_mode == "Country Comparison":
        # Multi-country comparison
        st.sidebar.markdown("### 🌍 Select Countries")
        available_countries = get_available_countries(df)

        selected_countries = st.sidebar.multiselect(
            "Choose countries to compare (max 8):",
            options=available_countries,
            default=["United States", "India", "Brazil"],
            max_selections=8,
        )

        if not selected_countries:
            st.warning("Please select at least one country for comparison.")
            return

        # Filter data for selected countries
        countries_data = {}
        for country in selected_countries:
            country_df = df[df["country"] == country].copy()
            country_df = country_df[
                (country_df["date"] >= start_date_dt)
                & (country_df["date"] <= end_date_dt)
            ]
            country_df.set_index("date", inplace=True)
            countries_data[country] = country_df

        st.markdown(f"## 🌍 Country Comparison: {', '.join(selected_countries)}")

        # Comparison chart
        st.markdown("### 📈 Cases Comparison")
        comparison_chart = create_comparison_chart(countries_data, selected_countries)
        st.plotly_chart(comparison_chart, use_container_width=True)

        # Heatmap
        st.markdown("### 🔥 Cases Heatmap")
        heatmap_chart = create_heatmap_chart(
            countries_data, selected_countries, start_date_dt, end_date_dt
        )
        if heatmap_chart:
            st.plotly_chart(heatmap_chart, use_container_width=True)
        else:
            st.warning("Not enough data for heatmap visualization.")

        # Rankings
        st.markdown("### 🏆 Country Rankings")
        ranking_metric = st.selectbox(
            "Rank by:",
            options=["total_cases", "max_cases", "avg_cases", "recent_activity"],
            format_func=lambda x: {
                "total_cases": "Total Cases",
                "max_cases": "Maximum Daily Cases",
                "avg_cases": "Average Daily Cases",
                "recent_activity": "Recent Activity (30-day avg)",
            }[x],
        )

        ranking_chart = create_ranking_chart(
            countries_data, selected_countries, ranking_metric
        )
        if ranking_chart:
            st.plotly_chart(ranking_chart, use_container_width=True)

        # Export options
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Comparison Chart"):
                img_bytes = export_chart_as_png(comparison_chart)
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="country_comparison.png",
                    mime="image/png",
                )

        with col2:
            if st.button("📥 Export All Data"):
                all_data = pd.concat(
                    [df for df in countries_data.values()],
                    keys=countries_data.keys(),
                    names=["Country"],
                )
                csv_data = export_data_as_csv(all_data, "comparison_data.csv")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="comparison_data.csv",
                    mime="text/csv",
                )

    elif analysis_mode == "Global Overview":
        # Global overview
        st.markdown("## 🌍 Global COVID-19 Overview")

        # Global statistics
        global_data = df[(df["date"] >= start_date_dt) & (df["date"] <= end_date_dt)]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_global_cases = global_data["new_cases_per_million"].sum()
            st.metric("Global Cases per Million", f"{total_global_cases:,.0f}")

        with col2:
            avg_global_cases = global_data["new_cases_per_million"].mean()
            st.metric("Average Daily Cases", f"{avg_global_cases:.1f}")

        with col3:
            max_global_cases = global_data["new_cases_per_million"].max()
            st.metric("Peak Daily Cases", f"{max_global_cases:.1f}")

        with col4:
            active_countries = global_data.groupby("country")[
                "new_cases_per_million"
            ].sum()
            active_countries = active_countries[active_countries > 0]
            st.metric("Active Countries", len(active_countries))

        # Top countries by total cases
        st.markdown("### 🏆 Top Countries by Total Cases")
        country_totals = (
            global_data.groupby("country")["new_cases_per_million"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig = go.Figure(
            data=go.Bar(
                x=country_totals.values,
                y=country_totals.index,
                orientation="h",
                marker_color="lightcoral",
            )
        )

        fig.update_layout(
            title="Top 10 Countries by Total Cases per Million",
            xaxis_title="Total Cases per Million",
            yaxis_title="Country",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Global timeline
        st.markdown("### 📈 Global Timeline")
        global_timeline = global_data.groupby("date")["new_cases_per_million"].sum()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=global_timeline.index,
                y=global_timeline.values,
                mode="lines",
                name="Global Daily Cases",
                line=dict(color="red", width=2),
                fill="tonexty",
            )
        )

        fig.update_layout(
            title="Global Daily COVID-19 Cases",
            xaxis_title="Date",
            yaxis_title="Cases per Million",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Regional analysis
        st.markdown("### 🌍 Regional Analysis")

        # Define regions
        regions = {
            "North America": ["United States", "Canada", "Mexico"],
            "Europe": [
                "Germany",
                "France",
                "Italy",
                "Spain",
                "United Kingdom",
                "Netherlands",
                "Belgium",
                "Sweden",
                "Austria",
                "Denmark",
                "Finland",
                "Norway",
                "Switzerland",
            ],
            "Asia": [
                "China",
                "India",
                "Japan",
                "South Korea",
                "Indonesia",
                "Thailand",
                "Singapore",
                "Malaysia",
                "Vietnam",
                "Pakistan",
                "Iran",
                "Philippines",
            ],
            "South America": ["Brazil", "Argentina", "Colombia", "Chile"],
            "Africa": ["South Africa", "Egypt"],
            "Oceania": ["Australia"],
        }

        region_data = {}
        for region, countries in regions.items():
            region_countries = [c for c in countries if c in df["country"].unique()]
            if region_countries:
                region_df = global_data[global_data["country"].isin(region_countries)]
                region_data[region] = region_df.groupby("date")[
                    "new_cases_per_million"
                ].sum()

        if region_data:
            fig = go.Figure()
            colors = px.colors.qualitative.Set3

            for i, (region, data) in enumerate(region_data.items()):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode="lines",
                        name=region,
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )

            fig.update_layout(
                title="Regional COVID-19 Trends",
                xaxis_title="Date",
                yaxis_title="Cases per Million",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div class="feature-box">
        <h4>🎯 Dashboard Features:</h4>
        <ul>
            <li><strong>Single Country Analysis:</strong> Deep dive into one country with detailed charts and predictions</li>
            <li><strong>Country Comparison:</strong> Compare up to 8 countries with heatmaps and rankings</li>
            <li><strong>Global Overview:</strong> Worldwide analysis with regional breakdowns</li>
            <li><strong>Interactive Charts:</strong> Hover for details, zoom, and pan</li>
            <li><strong>Export Options:</strong> Download charts as PNG and data as CSV</li>
            <li><strong>Search Function:</strong> Quickly find countries by typing</li>
            <li><strong>Date Range Filtering:</strong> Analyze specific time periods</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
