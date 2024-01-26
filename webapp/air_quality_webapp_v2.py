#run the script with:
#py -m streamlit run air_quality_webapp.py
#app will open in a new tab in default web browser


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from sklearn.model_selection import train_test_split
import seaborn as sns
from datetime import timedelta
from sklearn.impute import SimpleImputer
import pathlib
import pickle
import os
import seaborn as sns

# Function to create lag features for time series data
def create_lag_features(df, target_column, lag):
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in the DataFrame.")
    
    df_copy = df.copy()
    for i in range(1, lag + 1):
        df_copy[f"{target_column}_lag_{i}"] = df_copy[target_column].shift(i)
    return df_copy.dropna(subset=[f"{target_column}_lag_{lag}"])


path_data = '../data/data combined (aqcin)/combined aqicn data (all german cities).csv'
df_data_all_cities = pd.read_csv(path_data)
df_data = df_data_all_cities.copy()
df_data['date'] = pd.to_datetime(df_data['date'])
df_data.sort_values(by='date', ascending=True, inplace=True)
df_data.rename(columns={' pm10':'pm10', ' pm25':'pm25'}, inplace=True)
df_data.replace(' ', np.nan, inplace=True) 
df_data['pm10'] = pd.to_numeric(df_data['pm10']) # , downcast='float'
df_data['pm25'] = pd.to_numeric(df_data['pm25']) # , downcast='float'
df_data.replace({'munich':'Munich','berlin,-germany':'Berlin',
            'potsdam-zentrum,-germany':'Potsdam',
            'hannover,-germany':'Hannover',
            'stuttgart-bad-cannstatt, germany':'Stuttgart',
            'dresden-bergstraße,-germany':'Dresden',
            'sternschanze,-hamburg, germany':'Hamburg',
            'frankfurt-schwanheim,-germany':'Frankfurt',
            'kiel-bahnhofstr.-verk., schleswig-holstein, germany':'Kiel'}, inplace=True )


# drop nan values
df_data.dropna(axis = 0, inplace = True)

# convert string columns into numeric 
df_data['pm25'] = pd.to_numeric(df_data['pm25'], errors='coerce')
df_data['pm10'] = pd.to_numeric(df_data['pm10'], errors='coerce')

df_data.reset_index(drop = True, inplace = True)

# Create lag features
lag = 3  # the lag can be adjusted as needed
df_data_lagged_pm25 = create_lag_features(df_data, 'pm25', lag)


df_data_lagged_pm25.reset_index(drop = True, inplace = True)

temp_path = '../models'
pathlib.Path(temp_path).mkdir(parents=True, exist_ok=True)

# load the saved model
with open(os.path.join(temp_path, 'stacked_model_pm10.pkl'), 'rb') as model_file:
    stacked_model_pm10 = pickle.load(model_file)

# load the saved models
with open(os.path.join(temp_path, 'stacked_model_pm25.pkl'), 'rb') as model_file:
    stacked_model_pm25 = pickle.load(model_file)




# Function to make future predictions
def make_future_predictions(city, forecast_days, stacked_model):
    # Filter the complete data for the selected city
    complete_data_pm25_city = df_data[df_data['city'] == city]

    # Identify the last date in the complete data
    last_date_pm25 = complete_data_pm25_city['date'].max()

    # Use historical data up to the last available date for forecasting
    historical_data_pm25 = df_data[df_data['date'] <= last_date_pm25]

    # Create future dates for forecasting
    future_dates_pm25 = pd.date_range(last_date_pm25 + timedelta(days=1), periods=forecast_days, freq='D')

    # Create features for the future dates (lagged features based on historical data)
    future_features_pm25 = create_lag_features(pd.DataFrame({'date': future_dates_pm25, 'pm25': np.nan}), 'pm25', lag=lag).drop(columns=['date'])

    # Initialize or ensure 'features_pm' list exists
    features_pm25 = features_pm25 if 'features_pm25' in locals() else []

    # Include lagged features in the 'features' list without duplicates
    features_pm25 += [f"pm25_lag_{i}" for i in range(1, lag + 1) if f"pm25_lag_{i}" not in features_pm25]

    # Combine past and future features
    all_features_pm25 = pd.concat([create_lag_features(historical_data_pm25, 'pm25', lag)[features_pm25], future_features_pm25[features_pm25]], ignore_index=True)

    # Make predictions for both past and future using the stacked model
    all_predictions_pm25 = stacked_model.predict(all_features_pm25)

    return complete_data_pm25_city, future_dates_pm25, all_predictions_pm25[-forecast_days:]



# Suppress the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)




@st.cache_data
def load_data():
    df = pd.read_excel("cities.xlsx")
    df['date'] = pd.to_datetime(df['date'])
    df['pm10'] = pd.to_numeric(df['pm10'], downcast='float')
    df['pm25'] = pd.to_numeric(df['pm25'], downcast='float')
    df.set_index('city', inplace=True)
    return df

df = load_data()
city = st.multiselect('Choose a city/cities', pd.unique(df.index))
if not city :
    st.warning("Please select a city.")
    st.stop()
str1 = ", "
st.write('You selected: ', str1.join(city))
data = df.loc[city]

tab1, tab2, tab3 = st.tabs(["Info", "Historic", "Forecast"])

with tab1:
    st.subheader('Why is air quality important to us? :leaves:')
    st.text('One average person inhlaes 14000 liters of air every day.')
    st.text('Breathing clean air can lessen the possibility of disease from stroke,')
    st.text('heart disease, lung cancer as well as chronic and acute respiratory')
    st.text('illnesses such as asthma.')
    st.text('Studies show that bad air quality can increase the number of hospital')
    st.text('admissions and emergency department visits, school absences and lost work days.')
    st.text('Air quality effects our ecosystem. Air pollutants impairs plants grow')
    st.text('and animals health issues.')
    st.subheader('What is particular matter?')
    st.text('Particulate matter contains microscopic solids or liquid droplets that are')
    st.text('so small that they can be inhaled and cause serious health problems.')
    st.subheader('PM10 – harmful particulate matter :factory:')
    st.text('PM10 is a mixture of particles suspended in the air that do not exceed')
    st.text('10 micrograms in diameter. It is harmful because it contains benzopyrenes, furans,')
    st.text('dioxins and in short, carcinogenic heavy metals. PM10 air quality has a negative')
    st.text('effect on the respiratory system. It is responsible for coughing attacks, wheezing,')
    st.text('and the worsening of conditions for people with asthma or acute bronchitis.')
    st.subheader('PM2.5 – the most harmful pollution')
    st.text('PM2.5 are atmospheric aerosols with a maximum diameter of 2.5 micrometers.')
    st.text('This type of suspended particulate matter is considered the most dangerous to human')
    st.text('health. This is due to its very fine nature, and its ability to penetrate directly')
    st.text('into the bloodstream.')
    st.subheader('5 things you can do to reduce particular matter :sun_small_cloud:')
    st.text('1. Carpool, use public transportation, bike or walk everytime it is possible')
    st.text('2. Travel less or smart, avoid going by plane')
    st.text('3. Go lokal for buying cloth and food especially fruits and vegetables ')
    st.text('4. Avoid burning at home')
    st.text('5. Plant more trees and greenery')
    st.image('https://www.sustrans.org.uk/media/4343/4343.jpg?anchor=center&mode=crop&width=730&height=410')
    st.link_button("Get more information about air quality and pollution level :mag_right:", "https://aqicn.org/scale/")
    st.caption('References')
    st.caption('10 things you can do to help reduce air pollution today. (n.d.). Sustrans. https://www.sustrans.org.uk/our-blog/get-active/2020/in-your-community/10-things-you-can-do-to-help-reduce-air-pollution-today')
    st.caption('Air.(n.d.). Department of health. https://www.tn.gov/health/cedep/environmental/healthy-places/healthy-places/environmental-quality/eq/air.html#:~:text=Breathing%20clean%20air%20can%20lessen,long%2D%20and%20short%2Dterm.')
    st.caption('Why air quality matters. (2023, February 23). Ministry for the Environment. https://environment.govt.nz/facts-and-science/air/why-air-quality-matters/') 
    on = st.toggle('Let it snow :snowman:')
    if on:
        st.snow()

with tab2:
    st.header('Historic air quality data :open_book:')
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    
    st.subheader('air quality over time')
    for c in city :
        st.text(c)
        st.line_chart(data.loc[c], x="date", y=["pm10", "pm25"])


    st.subheader('_pm10_ pollution level')
    if len(city) == 1 :
        st.bar_chart(data['pm10_pollution_level'].value_counts())
    else :
        st.area_chart(data.groupby(['city']).pm10_pollution_level.value_counts().unstack())
        st.dataframe(data.groupby(['city']).pm10_pollution_level.value_counts().unstack())

    st.subheader('_pm25_ pollution level')
    if len(city) == 1 :
        st.bar_chart(data['pm25_pollution_level'].value_counts())
    else :
        st.area_chart(data.groupby(['city']).pm25_pollution_level.value_counts().unstack())
        st.dataframe(data.groupby(['city']).pm25_pollution_level.value_counts().unstack())

    h = """count - The number of not-empty values.
    mean - The average (mean) value.
    std - The standard deviation.
    min - the minimum value.
    25% - The 25% percentile.
    50% - The 50% percentile.
    75% - The 75% percentile.
    max - the maximum value."""
    st.subheader('_some statistics_', help=h)
    cols = st.columns(len(city))
    for c in city :
        cols[city.index(c)].text(c)
        cols[city.index(c)].write(data.loc[c][['pm10','pm25']].describe())
with tab3:
    st.header('Forecast based on machine learning :crystal_ball:')
    # Ask user for city selection
    city_options = df_data['city'].unique()
    selected_city = st.selectbox("Select a city:", city_options)

    # Select number of days to forecast
    forecast_days = st.slider('Select Number of Days to Forecast:', min_value=1, max_value=365, value=7)


    # Make future predictions
    complete_data_pm25_city, future_dates_pm25, future_predictions_pm25 = make_future_predictions(selected_city, forecast_days, stacked_model_pm25)

    # Plot past and forecasted values
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(complete_data_pm25_city['date'], complete_data_pm25_city['pm25'], label='Past PM25', marker='o')
    ax.plot(future_dates_pm25, future_predictions_pm25, label='Forecasted PM25', marker='o')

    ax.set_title(f'Past and Forecasted PM25 for {selected_city}')
    ax.set_xlabel('Date')
    ax.set_ylabel('PM25 Values')
    ax.legend()

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Display the forecasted values in a table
    forecast_table = pd.DataFrame({'Date': future_dates_pm25, 'Forecasted PM25': future_predictions_pm25})
    st.write('Forecasted PM25 Values:', forecast_table)


    