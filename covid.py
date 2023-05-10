#..Checkpoint Objective
#...You have been tasked to gather COVID-19 data from the API of your choice that contains both noise and valuable data. After that, you will clean and pre-process the data, perform exploratory data analysis (EDA) to gain insights, and select the best-suited supervised algorithm to predict the future number of cases. Finally, you will deploy the model using Streamlit.

#...Instructions
#..1. Choose a COVID-19 API of your choice that contains both valuable data and noise.
#..2. Use Python to gather the data from the API and store it in a Pandas DataFrame.
#...3. Clean the data by removing any irrelevant columns, null values, or duplicates.
#...4. Pre-process the data by normalizing and scaling the numerical data.
#...5. Perform EDA to identify trends, correlations, and patterns in the data. Use visualizations such as histograms, scatter plots, and heatmaps to help you understand the data better.
#...6. Choose the best-suited supervised algorithm to predict the future number of cases. Use techniques such as train-test split, cross-validation, and grid search to optimize the model's performance.
#...7. Once you have chosen the best-suited model, deploy it using Streamlit. Create a user-friendly interface that allows users to input data and view the model's predictions.
#...8. Deploy your streamlit app with streamlit share

#...Hints:
#...1. Use the requests library to fetch data from the API.
#...2. Use the Pandas library to store the data in a DataFrame and clean it.
#...3. Use the plotly  library to create visualizations.
#...4. Use the Scikit-learn library to build and optimize the model.
#...5. Use the Streamlit library to create the user interface.
 

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import date

st.set_page_config(page_title="Covid-19 Confirmed Cases Predictor", page_icon=":guardsman:", layout="wide")

st.title('Covid-19 Confirmed Cases Predictor')
st.markdown('This app predicts the value of covid-19 confirmed cases based on a trend analysis of data from inception of the pandemic to 2023')
st.image('covid.png')
st.write('COVID-19 is a highly infectious respiratory illness caused by the novel coronavirus SARS-CoV-2. It was first identified in Wuhan, China in December 2019 and has since spread to become a global pandemic. The virus is primarily spread through respiratory droplets when an infected person talks, coughs, or sneezes, and can also be transmitted by touching a surface contaminated with the virus and then touching one\'s face.\n Symptoms of COVID-19 can range from mild to severe and include fever, cough, shortness of breath, fatigue, loss of taste or smell, and body aches. Some people may experience no symptoms at all. \n COVID-19 can lead to severe respiratory illness, hospitalization, and death, especially in people with underlying health conditions or those over the age of 60. Vaccines are now available and effective in preventing severe illness and hospitalization.')

username = st.text_input('What is your name?')
button = st.button('Please click me to submit.')
if button:
    if username != '':
        st.markdown(f'Hello, {username}!')
    else:
        st.warning('Please input your username to continue.')

st.sidebar.write('Prediction Metrics')

year = st.sidebar.selectbox('What year do you want to predict?', (2020, 2021, 2022, 2023, 2024, 2025))

# Initialize geocoding API
country = st.sidebar.text_input('Enter the name of your desired country: ')
longitude, latitude = 0, 0

if country != '':
    url = f"https://nominatim.openstreetmap.org/search?q={country}&format=json"
    response = requests.get(url).json()
    latitude = float(response[0]['lat'])
    longitude = float(response[0]['lon'])

    long = st.sidebar.write(f'The longitude of your location is: {longitude}')

    lat = st.sidebar.write(f'The latitude of your location is: {latitude}')

predict_button = st.sidebar.button('Predict the number of confirmed cases')

# Load the pre-trained model
model = joblib.load(open('covid_pred.pkl', 'rb'))

# User input
input_variables = [[year, longitude, latitude]]
input_v = np.array(input_variables)

frame = ({'year': [year], 'longitude': [longitude], 'latitude': [latitude]})
st.write('These are your input variables: ')
frame = pd.DataFrame(frame)
frame = frame.rename(index={0: 'Value'})
frame = frame.transpose()
st.write(frame)

# Make prediction
if predict_button:
    regressor = model.predict(input_v)
    current_date = date.today()
    st.write(f'The estimated average number of confirmed cases in {country} in the year {year} is, {int(regressor[0])}')

