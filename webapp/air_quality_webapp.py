#run the script with:
#py -m streamlit run air_quality_webapp.py
#app will open in a new tab in default web browser

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

st.header(':blue[Air Quality] :fog:', divider='blue')

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
    days = st.slider('How many days should the forecast be for?', 1, 14, 7)
    st.write("You choose ", days, ' day/days.')
    st.subheader('Work in progress :construction:')
    

