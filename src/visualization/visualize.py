# Streamlit live coding script
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('src/data/marketing_campaign_cleaned.csv', index_col=[0])

st.title("Customer personality analysis")

st.markdown('## 1. Descriptive statistics')

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

st.subheader('Success of advertising campaings')
campaigns_df = pd.melt(df, value_vars=['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5', 'Response'], var_name='campaign', value_name='success', ignore_index=True)
success_df = campaigns_df[campaigns_df.success == 1]
fig = px.histogram(success_df, x='success', y='campaign', color='campaign')
st.plotly_chart(fig)

st.subheader('Amount spent by age group')


st.subheader('Number of customers per spent group')
spend_5_500 = df["Spent"][(df["Spent"] >=5) & (df["Spent"] <= 500)]
spend_501_1000 = df["Spent"][(df["Spent"] >=501) & (df["Spent"] <= 1000)]
spend_1001_1500 = df["Spent"][(df["Spent"] >=1001) & (df["Spent"] <= 1500)]
spend_1501_2000 = df["Spent"][(df["Spent"] >=1501) & (df["Spent"] <= 2000 )]
spend_2001_2525 = df["Spent"][(df["Spent"] >=2001) & (df["Spent"] <= 2525)]

spend_x = ['5-500', '501-1000', '1001-1500', '1501-2000', '2001-2525']
spend_y = [len(spend_5_500.values), len(spend_501_1000.values), len(spend_1001_1500.values), len(spend_1501_2000.values), len(spend_2001_2525.values)]
d = {'spent_group': spend_x, 'count': spend_y}
spent_df = pd.DataFrame(d)
fig = px.bar(spent_df, x='spent_group', y='count', color='spent_group')
fig.update_layout(
  xaxis={'title': {'text': 'Spent group'}},
  yaxis={'title': {'text': 'Number of customers'}}
)
st.plotly_chart(fig)

st.subheader('Number of customers per income group')
income_1_30k = df["Income"][(df["Income"] >=1000) & (df["Income"] <= 30000)]
income_30_60k = df["Income"][(df["Income"] >=30001) & (df["Income"] <= 60000)]
income_60_90k = df["Income"][(df["Income"] >=60001) & (df["Income"] <= 90000)]
income_90_120k = df["Income"][(df["Income"] >=90001) & (df["Income"] <= 120000 )]
income_120_170k = df["Income"][(df["Income"] >=120001) & (df["Income"] <= 170000)]

income_x = ['1k - 30k', '30k - 60k', '60k - 90k', '90k - 120k', '120k - 170k']
income_y = [len(income_1_30k.values), len(income_30_60k.values), len(income_60_90k.values), len(income_90_120k.values), len(income_120_170k.values)]
d = {'income_group': income_x, 'count': income_y}
income_df = pd.DataFrame(d)
fig = px.bar(income_df, x='income_group', y='count', color='income_group')
fig.update_layout(
  xaxis={'title': {'text': 'Income group'}},
  yaxis={'title': {'text': 'Number of customers'}}
)
st.plotly_chart(fig)

st.markdown('## 2. Inductive statistics')
st.markdown('''
### Is being a parent correlated to the ammount of spendings on all product categories?
- Random component - the distribution of Y (is parent) is binomial
- Systematic component: Xs (amount of products bought per category) are explanatory variables
- Link function: Logit
''')
x = [i for i in range(5)]
y = [1390, 1367, 1264, 1143, 1289]
fig = go.Figure(
  data = go.Scatter(x=x, y=y, marker={'size': 10})
)
fig.update_layout(
  xaxis={'title': {'text': 'Backward elimination iteration'}},
  yaxis={'title': {'text': 'AIC'}}
)
st.plotly_chart(fig)
image = Image.open('src/data/model_sum.png')
st.image(image)
st.markdown('''
The best model has shown that amount spent on meat, fish products, web and catalog purchases
and deals has highly significant p-value, hence there is a relationship between those and being a parent.
''')

st.markdown('### Predicting whether a client is a parent based on the amount they spent')
image = Image.open('src/data/parent_odds.png')
st.image(image)

st.markdown('## 3. Customer clustering')
image = Image.open('src/data/corr.png')
st.image(image)

st.markdown('''From this heatmap we can observe the following clusters of correlated features: 

The high income cluster:
  - Amount spent and number of purchases are positively correlated with income
  - Purchasing in store, on the web or via the catalog is postiveley correlated with income

The have kids & teens:
  - Amount spend and number of purchases are negatively correlated with children
  - Purchasing deals is positively correlated with children

The advertising campaigns:
  - Acceptance of the advertising campaigns are strongly correlated with each other
  - Weak positive correlation of the advertising campaigs is seen with the high income cluster, and weak negative correlation is seen with the have kids & teens cluster
''')