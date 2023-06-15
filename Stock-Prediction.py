import streamlit as st
from datetime import date 
import pandas as pd
import yfinance as yf
import plotly.express as px
from prophet import Prophet
# from plotly import graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly




START = "2017-01-01" 
TODAY= date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction App")
# displaying all the possible stock from the github repo 
stocks =pd.read_csv('https://raw.githubusercontent.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/master/Yahoo-Finance-Ticker-Symbols.csv')

selected_stocks = st.selectbox("Select Stock for Prediction",stocks)


@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data.........")
data = load_data(selected_stocks)
data_load_state.text("Data Loaded ... done ! ")

st.subheader('Stock data')
st.text("(All prices are in USD)")

st.text('Fixed width text')

st.write(data.tail())   # last 5 column will be printed of the selected stock


# #df = px.data.stocks()
fig = px.line(data, x='Date', y=['Open','Close'])

fig.layout.update(title_text = "Time Series Data" , xaxis_rangeslider_visible = True)

fig.update_layout(
    margin=dict(l=10, r=20, t=50, b=40),
    )
st.plotly_chart(fig)


n_years = st.slider("Years of Prediction: ", 1,5)    #sliding bar 
period = n_years*365

#Forecasting
df_train = data[['Date','Close']]
df_train  = df_train.rename(columns={"Date": "ds","Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Predict Stock')
st.write(forecast.tail())

st.write('Predict Stock')
# fig1 = go(m,forecast)
fig1 = plot_plotly(m, forecast )
st.plotly_chart(fig1)

st.write('forecast Components')
fig2 = plot_components_plotly(m, forecast)
st.write(fig2)