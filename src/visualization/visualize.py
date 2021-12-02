# Streamlit live coding script
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json

df = pd.read_csv('src/data/marketing_campaign_cleaned.csv')

st.title("Customer personality analysis")

st.sidebar.markdown('# Table of Contents')

st.markdown('## 1. Dataset')

st.markdown('## 2. Descriptive statistics')

st.subheader('Education')
fig = px.histogram(df, x='Education', color='Education')
st.plotly_chart(fig)

st.subheader('Income distribution')
fig = px.histogram(df, x='Income')
st.plotly_chart(fig)

st.subheader('Martial status')
fig = px.histogram(df, x='Living_With', color='Living_With')
st.plotly_chart(fig)

st.subheader('Education by martial status')
sunburst_df = df[['Education', 'Living_With']]
fig = px.sunburst(sunburst_df, path=['Living_With', 'Education'])
st.plotly_chart(fig)

st.subheader('Recency')
fig = px.histogram(df, x='Recency')
st.plotly_chart(fig)
