# COVID-19 Data Analysis and Visualization Project

## 🎯 Project Overview

This project provides an interactive COVID-19 data analysis dashboard using Python, pandas, numpy, matplotlib, seaborn, and Streamlit. The project works with real COVID-19 data from Our World in Data.

### 🧠 What You'll Learn

- **Data Loading & Cleaning**: Working with real-world COVID-19 data
- **Exploratory Analysis**: Understanding trends and patterns
- **Data Visualization**: Creating comprehensive charts and graphs
- **Statistical Analysis**: Calculating key metrics using NumPy
- **Interactive Dashboards**: Building web applications with Streamlit
- **Real-world Data Analysis**: Working with actual COVID-19 data

---

## 🚀 Quick Start Guide

### Step 1: Set Up Your Environment

1. **Install Python 3.8+** (if not already installed)
2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Launch the Dashboard

```bash
cd data
streamlit run covid_dashboard_complete.py
```

---

## 📁 Project Structure

```
COVIDScope/
├── data/
│   ├── covid_dashboard_complete.py    # Interactive Streamlit dashboard
│   └── owid-covid-data.csv.csv        # COVID-19 dataset
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## 🎨 Interactive Dashboard Features

### 📊 **Complete Dashboard** (`covid_dashboard_complete.py`)

- **Three Analysis Modes**:
  - **Single Country Analysis**: Detailed analysis for one country
  - **Country Comparison**: Compare up to 8 countries simultaneously
  - **Global Overview**: Worldwide COVID-19 trends
- **Enhanced Visualizations**:
  - Interactive line charts with hover details
  - Heatmaps for country comparison
  - Ranking charts by different metrics
  - Global trend analysis
- **Advanced Features**:
  - Multi-country selection with dropdowns
  - Real-time data filtering
  - Export functionality for charts and data
  - Comprehensive statistics tables
  - Prediction models with confidence metrics

### 🎯 **Dashboard Controls**

- **Country Selection**: Dropdown with countries having recent COVID activity
- **Date Range**: Predefined periods or custom date selection
- **Chart Selection**: Choose specific chart types or view all
- **Analysis Mode**: Switch between single country, comparison, or global view
- **Export Options**: Download charts as PNG and data as CSV

---

## 📊 Dataset Information

The project uses COVID-19 data from Our World in Data with the following structure:

- **Entity**: Country name
- **Day**: Date of the data
- **Daily new confirmed cases of COVID-19 per million people (rolling 7-day average, right-aligned)**: Daily cases per million

### 📈 Key Features of the Dataset

- **Global Coverage**: 249 countries and territories
- **Time Range**: 2020-01-09 to 2025-07-13
- **Data Points**: 498,219 total records
- **Metric**: Cases per million people (normalized for population comparison)
- **Smoothing**: 7-day rolling average for trend analysis

---

## 📊 Step-by-Step Implementation

### ✅ **Step 1: Import Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
```

### 📥 **Step 2: Load and Explore Data**

```python
# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv.csv")

# Basic exploration
print(f"Shape: {df.shape}")
print(f"Number of countries: {df['Entity'].nunique()}")
```

### 🧹 **Step 3: Data Cleaning**

```python
# Rename columns for easier use
df_clean = df.copy()
df_clean.columns = ['country', 'date', 'new_cases_per_million']

# Convert date column
df_clean['date'] = pd.to_datetime(df_clean['date'])
```

### 🎯 **Step 4: Country-Specific Analysis**

```python
# Filter for a specific country
target_country = 'United States'  # Change this to any country
country_df = df_clean[df_clean['country'] == target_country].copy()
country_df.set_index('date', inplace=True)
```

### 📈 **Step 5: Data Visualization**

```python
# Create comprehensive dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Daily Cases Per Million
axes[0, 0].plot(country_df.index, country_df['new_cases_per_million'])

# Cumulative Cases
country_df['cumulative_cases_per_million'] = country_df['new_cases_per_million'].cumsum()
axes[0, 1].plot(country_df.index, country_df['cumulative_cases_per_million'])

# Moving Averages
country_df['ma7'] = country_df['new_cases_per_million'].rolling(window=7).mean()
axes[1, 0].plot(country_df.index, country_df['ma7'])

# Weekly Summary
weekly_data = country_df['new_cases_per_million'].resample('W').mean()
axes[1, 1].bar(range(len(weekly_data)), weekly_data.values)

plt.show()
```

### 📊 **Step 6: Statistical Analysis**

```python
# Calculate basic statistics
new_cases = country_df['new_cases_per_million'].dropna()
mean_cases = np.mean(new_cases)
std_cases = np.std(new_cases)
max_cases = np.max(new_cases)
min_cases = np.min(new_cases)

print(f"Mean new cases per million per day: {mean_cases:.2f}")
print(f"Standard deviation: {std_cases:.2f}")
print(f"Maximum daily cases per million: {max_cases:.2f}")
print(f"Minimum daily cases per million: {min_cases:.2f}")
```

---

## 🎨 Visualizations Created

1. **📊 Daily Cases Per Million** - Daily new cases with 7-day smoothing
2. **📈 Cumulative Cases Per Million** - Total cases over time
3. **📉 Moving Averages Comparison** - 7-day vs 14-day trends
4. **📊 Weekly Summary** - Bar chart of weekly averages
5. **🔮 Prediction Chart** - Linear regression model with future predictions
6. **📈 Trend Analysis** - Growth rate and trend indicators
7. **📊 Growth Rate Analysis** - Daily percentage changes
8. **🌍 Country Comparison** - Multi-country line charts
9. **🔥 Heatmaps** - Visual comparison of countries over time
10. **📊 Ranking Charts** - Country rankings by different metrics

---

## 📈 Key Features

### ✅ **Data Analysis**

- **Multi-country support** - Analyze any country in the dataset
- **Time series analysis** - Handle date-based data properly
- **Missing value handling** - Robust data cleaning
- **Statistical summaries** - Comprehensive metrics

### 📊 **Visualizations**

- **Interactive charts** - Multiple plot types with Plotly
- **Moving averages** - Smooth trend lines
- **Color-coded data** - Easy interpretation
- **Professional styling** - Publication-ready graphics
- **Hover details** - Rich interactive information

### 📊 **Statistics**

- **Basic statistics** - Mean, median, std dev
- **Trend analysis** - Recent vs previous periods
- **Growth rate analysis** - Daily percentage changes
- **Weekly summaries** - Aggregated data

### 🌐 **Interactive Features**

- **Real-time filtering** - Dynamic data selection
- **Country comparison** - Side-by-side analysis
- **Export functionality** - Download charts and data
- **Responsive design** - Works on different screen sizes

---

## 🛠️ Customization Options

### Change Target Country

```python
# In the script, change this line:
target_country = 'United States'  # Change to any country name
```

### Available Countries with Recent Activity

The script automatically identifies countries with recent COVID activity:

- Sri Lanka
- United States
- And many more...

### Modify Analysis Period

```python
# Change the number of days for analysis:
recent_data = country_df[['new_cases_per_million']].dropna().tail(14)  # Change 14 to any number
```

---

## 📚 Learning Outcomes

### 🎯 **Technical Skills**

- **pandas**: Data manipulation and time series analysis
- **numpy**: Statistical computations and mathematical operations
- **matplotlib/seaborn**: Data visualization and charting
- **Streamlit**: Building interactive web applications
- **Plotly**: Creating interactive visualizations
- **Real-world Data Analysis**: Working with actual COVID-19 data

### 🧠 **Analytical Skills**

- **Data Exploration**: Understanding dataset structure
- **Trend Analysis**: Identifying patterns over time
- **Statistical Thinking**: Interpreting numerical results
- **Critical Evaluation**: Assessing model limitations
- **Interactive Design**: Creating user-friendly interfaces

---

## ⚠️ Important Notes

### 📊 **Data Disclaimer**

- This analysis uses historical COVID-19 data
- Predictions are for educational purposes only
- Real-world forecasting requires more sophisticated models
- Always consult official health sources for current information

### 🧠 **Model Limitations**

- Linear regression assumes linear trends
- Short-term predictions only (7 days)
- Does not account for external factors
- Educational demonstration, not medical advice

---

## 🎉 Project Completion Checklist

- [x] ✅ **Environment Setup** - Python and libraries installed
- [x] ✅ **Data Loading** - COVID-19 dataset loaded successfully
- [x] ✅ **Data Cleaning** - Missing values and date formatting handled
- [x] ✅ **Country Selection** - Target country data filtered
- [x] ✅ **Visualizations Created** - All charts generated
- [x] ✅ **Statistics Calculated** - Key metrics computed
- [x] ✅ **Results Interpreted** - Insights and trends identified
- [x] ✅ **Documentation Complete** - Code and results documented
- [x] ✅ **Interactive Dashboard** - Streamlit web application created
- [x] ✅ **Advanced Features** - Country comparison and export functionality
- [x] ✅ **User Interface** - Dropdowns, buttons, and interactive controls

---

## 📞 Support & Resources

### 🔗 **Useful Links**

- [Our World in Data COVID-19](https://ourworldindata.org/covid-cases)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [matplotlib Tutorial](https://matplotlib.org/stable/tutorials/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

### 📚 **Learning Resources**

- **Data Science Fundamentals** - Basic concepts and techniques
- **Time Series Analysis** - Working with temporal data
- **Statistical Modeling** - Building predictive models
- **Data Visualization** - Creating effective charts
- **Web Development** - Building interactive applications

---

## 🎯 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit dashboard
cd data && streamlit run covid_dashboard_complete.py
```

---

**🎯 Ready to start? Launch the interactive dashboard and explore the fascinating world of COVID-19 data analysis with interactive visualizations!**
