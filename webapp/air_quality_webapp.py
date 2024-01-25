#run the script with:
#py -m streamlit run air_quality_webapp.py
#app will open in a new tab in default web browser

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

st.header(':blue[air quality] :fog:', divider='blue')

@st.cache_data
def load_data():
    df = pd.read_excel("cities.xlsx")
    df['date'] = pd.to_datetime(df['date'])
    df['pm10'] = pd.to_numeric(df['pm10'], downcast='float')
    df['pm25'] = pd.to_numeric(df['pm25'], downcast='float')
    df.set_index('city', inplace=True)
    return df

try:
    df = load_data()
    city = st.multiselect('Choose a city/cities', pd.unique(df.index))
    str1 = ", "
    st.write('You selected: ', str1.join(city))
    data = df.loc[city]

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    
    st.subheader('air quality over time')
    for c in city :
        st.text(c)
        st.line_chart(data.loc[c], x="date", y=["pm10", "pm25"])
    st.caption('PM10 – harmful particulate matter')
    st.caption('PM10 is a mixture of particles suspended in the air that do not exceed 10 micrograms in diameter. It is harmful because it contains benzopyrenes, furans, dioxins and in short, carcinogenic heavy metals. PM10 air quality has a negative effect on the respiratory system. It is responsible for coughing attacks, wheezing, and the worsening of conditions for people with asthma or acute bronchitis.')
    st.caption('PM2.5 – the most harmful pollution')
    st.caption('PM2.5 are atmospheric aerosols with a maximum diameter of 2.5 micrometers. This type of suspended particulate matter is considered the most dangerous to human health. This is due to its very fine nature, and its ability to penetrate directly into the bloodstream.')
    
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
    
    st.link_button("Get more information about air quality and pollution level :mag_right:", "https://aqicn.org/scale/")
    
    on = st.toggle('Let it snow :snowman:')
    if on:
        st.snow()
except:
    st.error("Please select a city.")
    st.stop()