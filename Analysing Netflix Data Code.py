#!/usr/bin/env python
# coding: utf-8

# # Analysing Netflix Trends

# #### Krishnamohan Pingali (501130910) 

# In[1]:


# Importing necessary libraries for analysis and visualization purposes.

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Data Loading: Loading the dataset (netflix_titles.csv) from the local machine and making it more presentable by renaming a few attributes and providing delimiter options.


df = pd.read_csv("netflix_titles.csv", delimiter=",", encoding="latin-1", parse_dates=["date_added"], index_col=["show_id"])
df


# In[3]:


#Data Cleaning: Converting 'date_added' attribute format to better suit the research questions analysed by removing the month and date and keeping the year. 

df["date_added"] = df["date_added"].dt.year
df["date_added"].unique()


# In[4]:


#Data Cleaning

df["date_added"].fillna(0, inplace=True)
df["date_added"] = df["date_added"].astype(int)
df.head()


# In[5]:


# Checking for duplicate rows. None present in this dataset.

df.duplicated().sum()


# In[6]:


# Replacing all Null Values with 'No Data' for better analysis.

df.fillna("No Data", inplace=True)
df.isnull().sum()


# In[7]:


# Checking the data types of all attributes.

df.dtypes


# In[8]:


# Data Description: Summary statistics.

df.describe()


# In[9]:


# Data Description: Outlines the total numebr of entries (8807) and attribute names.

df.info()


# In[10]:


# Data Description: Brief view of the first 5 entries with all their details from the dataset.

df.head()


# ## Media Comparison

# In[11]:


# Using a pie chart to represent the distribution oftv shows to movies.

plt.figure(figsize=(14, 7))
labels=['TV Show', 'Movie']
plt.pie(df['type'].value_counts().sort_values(),labels=labels,explode=[0.1,0.1],
        autopct='%1.2f%%',colors=['skyblue','crimson'], startangle=90)
plt.title('Type of Media')
plt.axis('equal')
plt.show()


# In[12]:


# Using a bar graph to show the total entries of both movies and tv shows.

ax = sns.countplot(x="type", data=df)


# ## Media Count by Year

# In[13]:


# Using a stacked bar graph to represent the media added per each year starting from 2011 to 2021.


df_movies = df.loc[df['release_year'] > 2010]
agg_tips = df_movies.groupby(['release_year','type'])['type'].count().unstack()

fig, ax = plt.subplots(figsize=(14, 7))

colors = ['#24b1d1', '#ae24d1']
bottom = np.zeros(len(agg_tips))

for i, col in enumerate(agg_tips.columns):
  ax.bar(
      agg_tips.index, agg_tips[col], bottom=bottom, label=col, color=colors[i])
  bottom += np.array(agg_tips[col])

totals = agg_tips.sum(axis=1)
y_offset = 4
for i, total in enumerate(totals):
  ax.text(totals.index[i], total + y_offset, round(total), ha='center',
          weight='bold')

y_offset = -15
for bar in ax.patches:
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + bar.get_y() + y_offset,
      round(bar.get_height()),
      ha='center',
      color='w',
      weight='bold',
      size=8
  )

ax.set_title('Movie & TV Shows Count by Year')
ax.set_xlabel('Release Year')
ax.set_ylabel('Count')
ax.legend()


# ## Media produced per Country

# In[14]:


# Using a plotly pie chart to show the distribution of media produced by each country. Hovering over each portion of the pie reveals the total number of media pro


country_df = df['country'].value_counts().reset_index()
country_df = country_df[country_df['country'] /  country_df['country'].sum() > 0.01]

fig = px.pie(country_df, values='country', names='index', title="Media per Country")
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[15]:


# Comparing amount of movies and tv shows added since 2008 using a line graph to depict each movie and tv show added.

df = df.reset_index()
released_year_df = df.loc[df['release_year'] > 2000].groupby(['release_year', 'type']).agg({'show_id': 'count'}).reset_index()
added_year_df = df.loc[df['date_added'] > 2000].groupby(['date_added', 'type']).agg({'show_id': 'count'}).reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter( 
    x=released_year_df.loc[released_year_df['type'] == 'Movie']['release_year'], 
    y=released_year_df.loc[released_year_df['type'] == 'Movie']['show_id'],
    mode='lines+markers',
    name='Movie: Released Year',
    marker_color='orange',
))
fig.add_trace(go.Scatter( 
    x=released_year_df.loc[released_year_df['type'] == 'TV Show']['release_year'], 
    y=released_year_df.loc[released_year_df['type'] == 'TV Show']['show_id'],
    mode='lines+markers',
    name='TV Show: Released Year',
    marker_color='red',
))
fig.add_trace(go.Scatter( 
    x=added_year_df.loc[added_year_df['type'] == 'Movie']['date_added'], 
    y=added_year_df.loc[added_year_df['type'] == 'Movie']['show_id'],
    mode='lines+markers',
    name='Movie: Year Added',
    marker_color='yellow',
))
fig.add_trace(go.Scatter( 
    x=added_year_df.loc[added_year_df['type'] == 'TV Show']['date_added'], 
    y=added_year_df.loc[added_year_df['type'] == 'TV Show']['show_id'],
    mode='lines+markers',
    name='TV Show: Year Added',
    marker_color='greenyellow',
))
fig.update_xaxes(categoryorder='total descending')
fig.show()


# In[16]:


# Finding out the number of entries in each Media Rating.

df_movies['rating'].value_counts()


# ## Movie Ratings by Country

# In[17]:


# Using stacked bar graphs to determine the Movie Ratings per country.
df_movies = df.loc[df['type'] == 'Movie']
df2 = df_movies.loc[(df_movies['rating'] == 'TV-MA') | (df_movies['rating'] == 'R') |
                    (df_movies['rating'] == 'TV-14')| (df_movies['rating'] == 'TV-PG')]
df3 = df2.loc[(df2['country'] == 'United States')|(df2['country'] == 'India')|(df2['country'] == 'United Kingdom')|
              (df2['country'] == 'Japan')|(df2['country'] == 'South Korea')|(df2['country'] == 'Canada')
             |(df2['country'] == 'Spain')|(df2['country'] == 'France')|(df2['country'] == 'Mexico')]

agg_tips3 = df3.groupby(['country','rating'])['rating'].count().unstack()
agg_tips3.plot(kind='bar', stacked=True,figsize=(14, 7))
plt.title('Count of Movie Ratings based on Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

#


# In[18]:


# Using stacked bar graphs to determine the Movie Ratings per country.

df_movies = df.loc[df['type'] == 'Movie']
df2 = df_movies.loc[(df_movies['rating'] == 'TV-MA') | (df_movies['rating'] == 'R') |
                    (df_movies['rating'] == 'TV-14')| (df_movies['rating'] == 'TV-PG')]
df3 = df2.loc[(df2['country'] == 'United Kingdom')|
              (df2['country'] == 'Japan')|(df2['country'] == 'South Korea')|(df2['country'] == 'Canada')
             |(df2['country'] == 'Spain')|(df2['country'] == 'France')|(df2['country'] == 'Mexico')]

agg_tips3 = df3.groupby(['country','rating'])['rating'].count().unstack()
agg_tips3.plot(kind='bar', stacked=True,figsize=(14, 7))
plt.title('Count of Movie Ratings based on Country Excluding India and United States')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')


# In[ ]:




