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
category = pd.cut(df.Age,bins=[25,40,65,81],labels=['Adults','Seniors', 'Elderly'])
df.insert(2,'Age_Groups',category)
mean_spent = df.groupby(['Age_Groups']).Spent.mean().reset_index()
fig = px.bar(mean_spent, x='Age_Groups', y='Spent', color='Age_Groups')
fig.update_layout(
  xaxis=dict(
    title={'text': ''},
    tickvals=[0, 1, 2],
    ticktext=['25-40', '40-65', '65+']
  ),
  yaxis={'title': {'text': 'Mean spendings'}}
)
st.plotly_chart(fig)

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

st.subheader('Does age vary across customer that accepted the campaing?')
x = [1, 2, 3, 4, 5, 6]
y = [0.009691423054819294, 0.8935207294191613, 0.008222422455038243, 0.08503414438616336, 1.1933101456748575e-05, 0.009158292100321172]
fig = px.scatter(x=x, y=y, color_discrete_sequence=['yellow'])
fig.update_traces(marker={'size': 16})
fig.update_layout(
  xaxis={'title': {'text': 'Number of advertising campaing'}},
  yaxis={'title': {'text': 'P-value'}}
)
fig.add_hrect(y0=1.2, y1=0.05, line_width=0, fillcolor="red", opacity=0.2, annotation_text='rejection region', annotation=dict(font_size=16))
st.plotly_chart(fig)

x=['1st campaign', '2nd campaign', '3rd campaign', '4th campaign', '5th campaign', '6th campaign' ]
fig = go.Figure(go.Bar(x=x, y=[8.5, 1.3, 11.1, 5.8, 12.2, 19.8], name='25 - 40'))
fig.add_trace(go.Bar(x=x, y=[5.3, 1.4, 6.5, 7.2, 5.5, 13.6], name='40 - 65'))
fig.add_trace(go.Bar(x=x, y=[8.9, 1.1, 6.7, 10,9.4, 15.8], name='Above 65'))
fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})
st.plotly_chart(fig)
st.markdown('''
We can conclude that the 1st, 3rd , 5th and the last campaign are dependent on the age groups. Althoght we dont know the exact kind of the campaign, statistically we can say that these 4 campaigns especially the last campaign was targeted for specific age group. From the chart below we can conclude that most of the campaign focuses on the age group from 25 to 40 and above 65. the reason could be to motivate this age groups , since most of the customers are not on these two age groups but on the age group 40 to 65. Also we can say that: regardless of the age group , the customer that have accepted the offer in the 2nd campaign is much lower and the customer have accepted the offer in the responce (last campaign) is better than the other. 
''')

st.markdown('### Spendings after last advertising campaing')
st.markdown('We used kruskalwallis test to test whether customers that accepted the last camaping spent more. Resulting p-value was equal to 1')

image = Image.open('src/data/box.png')
st.image(image)

image = Image.open('src/data/camp_spend.png')
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