import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("GlobalLandTemperaturesByState.csv")
    df = df.dropna(how='any', axis=0)
    df.rename(columns={'dt':'Date', 'AverageTemperature':'avg_temp', 'AverageTemperatureUncertainty':'confidence_interval_time'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year #type:ignore
    
    return df

df = load_data()

latest_df = df.loc['1975':'2013']
resample_df = latest_df[['avg_temp']].resample('A').mean()

# Streamlit app layout
st.title('Climate Change Data Analysis')
st.write('An interactive web app to analyze climate changes from 1975 to 2013.')

# Sidebar for user selection
option = st.sidebar.selectbox(
    'Select visualization',
    ['Line Plot', 'Trend-Seasonal Decomposition', 'Rolling Statistics', 'ACF and PACF']
)

if option == 'Line Plot':
    st.subheader('Temperature Changes from 1975-2013')
    
    plt.figure(figsize=(9,4))
    
    sns.lineplot(x="Year", y="avg_temp", data=latest_df)
    
    plt.title('Temperature Changes from 1975-2013')
    plt.xlabel('Year')
    plt.ylabel('Temperature')
    
    st.pyplot(plt) #type:ignore
    st.write("**Description**: This line plot shows the average temperature changes from 1975 to 2013. It helps in visualizing the overall trend of temperature variations over the years.")

elif option == 'Trend-Seasonal Decomposition':
    st.subheader('Trend, Seasonal, and Residual Decomposition')
    
    decomp = seasonal_decompose(resample_df, period=3)
    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.resid
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    
    ax1.plot(resample_df, label='Original')
    ax1.set_ylabel('Original')
    ax2.plot(trend, label='Trend')
    ax2.set_ylabel('Trend')
    ax3.plot(seasonal, label='Seasonal')
    ax3.set_ylabel('Seasonal')
    ax4.plot(residual, label='Residual')
    ax4.set_ylabel('Residual')
    
    fig.tight_layout()
    
    st.pyplot(fig)
    st.write("**Description**: This decomposition plot breaks down the time series into three components: Trend, Seasonal, and Residual. It helps in understanding the underlying patterns and seasonality in the temperature data.")

elif option == 'Rolling Statistics':
    st.subheader('Rolling Mean and Standard Deviation')
    rol_mean = resample_df.rolling(window=3, center=True).mean()
    ewm = resample_df.ewm(span=3).mean()
    rol_std = resample_df.rolling(window=3, center=True).std()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(resample_df, label="Original")
    ax1.plot(rol_mean, label="Rolling Mean")
    ax1.plot(ewm, label="Exponentially Weighted Mean")
    ax1.set_title('Temperature Changes from 1975-2013')
    ax1.set_ylabel("Temperature")
    ax1.set_xlabel("Year")
    ax1.legend()

    ax2.plot(rol_std, label="Rolling Standard Deviation")
    ax2.set_title('Rolling Standard Deviation')
    ax2.set_ylabel("Temperature")
    ax2.set_xlabel("Year")
    ax2.legend()
    
    st.pyplot(fig)
    st.write("**Description**: This plot shows the rolling mean and rolling standard deviation of the temperature data. The rolling statistics help in identifying trends and the stability of the time series over the specified window.")

elif option == 'ACF and PACF':
    st.subheader('Autocorrelation and Partial Autocorrelation Functions')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    plot_acf(resample_df, ax=ax1)
    plot_pacf(resample_df, ax=ax2)
    
    st.pyplot(fig)
    st.write("**Description**: The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots help in understanding the correlation of the time series with its own past values.")

# Display Dickey-Fuller Test Results
st.write("### Dickey-Fuller Test Results")

test_df = adfuller(resample_df.iloc[:, 0].values, autolag='AIC')
df_output = pd.Series(test_df[0:4], index=['Test Statistics', 'p-value', 'Lags Used', 'Number of Observations'])

for key, value in test_df[4].items(): #type:ignore
    df_output['Critical Value (%s)' % key] = value

st.write(df_output)
st.write("**Description**: The Dickey-Fuller test checks the stationarity of the time series data. A low p-value (< 0.05) indicates that the time series is stationary, meaning it has a constant mean and variance over time.")
