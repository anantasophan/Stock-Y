import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import plotly.express as px
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gnews import GNews
from newspaper import Article
import datetime
import requests
from bs4 import BeautifulSoup
from PIL import Image
image1 = Image.open('bca.png')
image2 = Image.open('mandiri.png')
image3 = Image.open('bni.png')
image4 = Image.open('bri.png')
image5 = Image.open('btn.png')
logo = Image.open('STOCK-removebg.png')

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

def news(selected_stock):
  st.header("Headline News")
  if selected_stock == 'BBRI':
    url = "https://www.google.com/search?sxsrf=ALiCzsaW9Ve40dpjT2L2MbzLZSyj0UfoYA:1671421151309&q=bbri+saham&tbm=nws&source=univ&tbo=u&sxsrf=ALiCzsaW9Ve40dpjT2L2MbzLZSyj0UfoYA:1671421151309&sa=X&ved=2ahUKEwixqvn_4IT8AhW5TWwGHbb0B_AQt8YBKAB6BAgXEAE&biw=1536&bih=714&dpr=1.25"
  elif selected_stock == 'BBCA':
    url = "https://www.google.com/search?q=bbca+saham&biw=1536&bih=714&tbm=nws&sxsrf=ALiCzsYZF65FYfcDywaK6tqFhgQoX_4C2Q%3A1671435753828&ei=6RWgY7qVMt2n3LUPlKOVmAI&ved=0ahUKEwj6rfyyl4X8AhXdE7cAHZRRBSMQ4dUDCA0&uact=5&oq=bbca+saham&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzIECAAQQzIFCAAQgAQyBQgAEIAEMgQIABBDMgQIABBDMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIGCAAQBxAeOgoIABCxAxCDARBDOgYIABAWEB46CAgAEAgQBxAeOgcIABCxAxBDOgsIABCABBCxAxCDAToKCAAQgAQQsQMQCjoHCAAQgAQQCjoHCAAQgAQQDVCjHFjsJWDSK2gAcAB4AIAB0gGIAf8HkgEFNS4zLjGYAQCgAQHAAQE&sclient=gws-wiz-news"
  elif selected_stock == 'BBNI':
    url = "https://www.google.com/search?q=bbni+saham&biw=1536&bih=714&tbm=nws&sxsrf=ALiCzsaT9AndHtEm-mKIFpFeUriuDnPI5A%3A1671435851611&ei=SxagY8_qJPPnz7sPhM2uGA&ved=0ahUKEwiPvMzhl4X8AhXz83MBHYSmCwMQ4dUDCA0&uact=5&oq=bbni+saham&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzILCAAQgAQQsQMQgwEyBQgAEIAEMgYIABAHEB4yBggAEAcQHjIFCAAQgAQyBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB46DQgAEIAEELEDEIMBEAo6BwgAEIAEEAo6CAgAEAcQHhAKUKgMWKgMYPgQaAFwAHgAgAFNiAHLAZIBATOYAQCgAQHAAQE&sclient=gws-wiz-news"
  elif selected_stock == 'BBTN':
    url = "https://www.google.com/search?q=bbtn+saham&biw=1536&bih=714&tbm=nws&sxsrf=ALiCzsZJPwVQ0WJor-EFo7dY-IB9FaryuQ%3A1671435863728&ei=VxagY6DSK46A3LUPtvihmAQ&ved=0ahUKEwjg2a_nl4X8AhUOALcAHTZ8CEMQ4dUDCA0&uact=5&oq=bbtn+saham&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzINCAAQgAQQsQMQgwEQDTIECAAQQzIGCAAQBxAeMgQIABAeMgQIABAeMgQIABAeMgYIABAFEB4yBggAEAgQHjIGCAAQCBAeMgYIABAIEB46CggAELEDEIMBEEM6CwgAEIAEELEDEIMBOgUIABCABDoICAAQCBAHEB46CggAEIAEELEDEAo6BwgAEIAEEAo6BwgAEIAEEA06CAgAEAcQHhAKOgoIABAIEAcQHhAKOg8IABCABBCxAxCDARANEApQ5wZYkhRgkBZoAnAAeACAAXSIAYwGkgEDNi4zmAEAoAEBwAEB&sclient=gws-wiz-news"
  else:
    url = "https://www.google.com/search?tbm=nws&sxsrf=ALiCzsZtLZUC7bGTbjg_NdM6EDljm8dkBQ:1671435919925&q=bmri+saham&spell=1&sa=X&ved=2ahUKEwjuiZaCmIX8AhXKnNgFHXUYBQgQBSgAegQIBxAB&biw=1536&bih=714&dpr=1.25"

  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
  page = requests.get(url, headers=headers)
  soup = BeautifulSoup(page.content, "lxml")

  # Find all heading elements
  heading_elements = soup.find_all('div', class_='mCBkyc ynAwRc MBeuO nDgy9d')

  link_elements = soup.find_all('a', class_='WlydOe')

  # Loop through each link element and store the link


  for link_element in link_elements:
    link = link_element.get('href')
    headline =  link_element.get_text()
    st.markdown(f'<p class="big-font">{headline}</p>', unsafe_allow_html=True)
    st.markdown(link, unsafe_allow_html=True)
    st.markdown("""<hr style="height:3px;border:none;color:#49403C;background-color:#49403C;" /> """, unsafe_allow_html=True)

def suggest(A, S):

  S = float(S)
  if A == 'Yes':
    if S <= -1.0:
      st.header('Sell :cry:') 
      st.markdown(f'<p class="big-font">Because for the next 30 days will going down more than 1% </p>', unsafe_allow_html=True)
    elif S >= 1.0:
      st.header('Buy More :money_mouth_face:')
      st.markdown(f'<p class="big-font">Because for the next 30 days will going up more than 1% </p>', unsafe_allow_html=True)
    else:
      st.header('Hold :sunglasses:')
      st.markdown(f'<p class="big-font">Because for the next 30 days The price do not have significant changes </p>', unsafe_allow_html=True)
  
  else:
    if S <= -1.0:
      st.header('Dont Buy :cry:')
      st.markdown(f'<p class="big-font">Oops! The stock you choose might be going down in the next 30 days</p>', unsafe_allow_html=True)
    elif S >= 1.0:
      st.header('Buy :money_mouth_face:')
      st.markdown(f'<p class="big-font">Time to buy! Estimatedly this stock will go up in the next 30 days</p>', unsafe_allow_html=True)
    else:
      st.header('Up to you :stuck_out_tongue_winking_eye:')
      st.markdown(f'<p class="big-font">Make your own decision, this stock price will be stagnant for the next 30 days</p>', unsafe_allow_html=True)


# Load the models and scalers
scaler_bca = pickle.load(open('scaler_bca.pkl','rb'))
model_bca = tf.keras.models.load_model('model_bca.h5')
scaler_bmri = pickle.load(open('scaler_bmri.pkl','rb'))
model_bmri = tf.keras.models.load_model('model_bmri.h5')
scaler_bni = pickle.load(open('scaler_bni.pkl','rb'))
model_bni = tf.keras.models.load_model('model_bni.h5')
scaler_bri = pickle.load(open('scaler_bri.pkl','rb'))
model_bri = tf.keras.models.load_model('model_bri.h5')
scaler_btn = pickle.load(open('scaler_btn.pkl','rb'))
model_btn = tf.keras.models.load_model('model_btn.h5')

# Dictionary to map stock names to ticker symbols and models/scalers
stock_data = {
    'BBCA': {'ticker': 'BBCA.JK', 'model': model_bca, 'scaler': scaler_bca, 'logo': image1},
    'BMRI': {'ticker': 'BMRI.JK', 'model': model_bmri, 'scaler': scaler_bmri, 'logo': image2},
    'BBNI': {'ticker': 'BBNI.JK', 'model': model_bni, 'scaler': scaler_bni,'logo': image3},
    'BBRI': {'ticker': 'BBRI.JK', 'model': model_bri, 'scaler': scaler_bri,'logo': image4},
    'BBTN': {'ticker': 'BBTN.JK', 'model': model_btn, 'scaler': scaler_btn,'logo': image5},
}

# Set the start and end dates for the historical data
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the time step for the model
time_step = 8

# Set up the web app
st.image(logo)

# Allow the user to select a stock and specify the number of days to predict
stocks = ('BBCA', 'BMRI', 'BBNI', 'BBRI','BBTN')
selected_stock = st.sidebar.selectbox('Stocks', stocks)
n_days = st.sidebar.slider('Days of prediction:', 7, 30)

# Display the historical data for the selected stock

st.subheader('Historical Stock Report')
st.image(stock_data[selected_stock]['logo'],width=100)
# Get the data for the selected stock

data = yf.download(stock_data[selected_stock]['ticker'], START, TODAY)
data = data[["Open", "High","Low", "Close","Volume"]]
st.write(data.tail())

Stock = st.sidebar.radio(
    "Do you already have the stock?",
    ('Yes', 'No'))


if st.sidebar.button("Predict"):
  # Preprocess the data
  data.reset_index(inplace=True)
  data = data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                              'Adj Close':'adj_close','Volume':'volume'})
  data['date'] = pd.to_datetime(data.date)
  data = data['close']
  input_data = stock_data[selected_stock]['scaler'].transform(np.array(data).reshape(-1,1))

  # Create the dataset for the model
  def create_dataset(dataset, time_step=1):
      dataX, dataY = [], []
      for i in range(len(dataset)-time_step-1):
          a = dataset[i:(i+time_step), 0]
          dataX.append(a)
          dataY.append(dataset[i + time_step, 0])
      return np.array(dataX), np.array(dataY)

  # Split the data into test and train sets
  X_test, y_test = create_dataset(input_data, time_step)
  X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

  # Make predictions using the model
  y_predicted = stock_data[selected_stock]['model'].predict(X_test)

  # Inverse transform the predictions and original data
  y_predicted = stock_data[selected_stock]['scaler'].inverse_transform(y_predicted)
  y_test = stock_data[selected_stock]['scaler'].inverse_transform(y_test.reshape(-1,1))

  # Plot the predictions vs the original data
  st.subheader("Predictions vs Original")
  fig2= plt.figure(figsize = (12,6))
  plt.plot(y_test, 'b', label = 'Original Price')
  plt.plot(y_predicted, 'r', label = 'Predicted Price')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend()
  st.pyplot(fig2)

  # Display the prediction for the specified number of days
  st.markdown(f'<p class="big-font">Prediction for the next {n_days} days</p>', unsafe_allow_html=True)
  y_predicted = y_predicted.reshape(-1)
  times = list(range(1, n_days+1))
  df = pd.DataFrame({'Day': times, 'Price': y_predicted[-n_days:]})
  fig = px.line(df, x='Day', y='Price', hover_name='Price')
  fig.update_traces(line_color='#ffff00', line_width=5)
  

  #Percentage for predictions
  L = y_predicted[-1]
  F = y_predicted[-n_days]
  P = round(((L-F)/F),2)
  P = '{:.2f}'.format(P)
  T = '{:,.2f}'.format(L)
  st.metric(stock_data[selected_stock]['ticker'], f'IDR {T}', f'{P}%')
  st.plotly_chart(fig)

  #Precentage for suggestion
  X = y_predicted[-30]
  X = round(((L-X)/X),2)
  X = '{:.2f}'.format(X)


  st.markdown(f'<p class="big-font">Our Suggestion</p>', unsafe_allow_html=True)
  suggest(Stock, X)



  # Display News Based on selected stock
  news(selected_stock)
