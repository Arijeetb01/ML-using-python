#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np


# In[166]:


movie_df = pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files)/Data/bollywood.csv')


# In[167]:


movie_df


# In[168]:


# How many records are present in dataset

movie_df.info()


# In[169]:


# How many movies got released in each genre ? 

movie_df.Genre.value_counts().reset_index()


# In[170]:


movie_df.Genre.value_counts(normalize= True)


# In[171]:


movie_df.Genre.value_counts(normalize= True)*100


# In[172]:


# Which genre has highest number of release times like long weekend, Festive season etc.

pd.crosstab(movie_df['Genre'],movie_df['ReleaseTime'])


# In[139]:


# Which month of the year typically sees most release of high budgeted movies, i.e. - budget > 25 Cr.


# In[214]:


# Creating New Column (month)

movie_df['month']= pd.to_datetime(movie_df['Release Date']).dt.month
movie_df_month=movie_df[['MovieName','month','Budget','ReleaseTime','Genre']]


# In[215]:


high_budgeted_movies=movie_df_month[movie_df['Budget']>25][['month','Budget']]
high_budgeted_movies


# In[216]:


most_release_of_high_budgeted_movies=high_budgeted_movies.month.value_counts()
most_release_of_high_budgeted_movies


# In[217]:


# Which month of the year,maximum number movies releases are seen?
maximum_number_movies_releases_seen = movie_df_month.month.value_counts()
maximum_number_movies_releases_seen


# In[218]:


# Highest ROI

movie_df_month['ROI']=movie_df.apply(lambda rec:(rec.BoxOfficeCollection-rec.Budget)/rec.Budget,axis=1)
movie_df_month


# In[219]:


movie_df_month.sort_values('ROI',ascending=False)[0:10]


# In[253]:


# Effect of Release Time on ROI

Effect_of_release_time_on_ROI = movie_df_month.groupby('ReleaseTime')['ReleaseTime','ROI'].mean()
Effect_of_release_time_on_ROI 


# In[ ]:


# Observations - 

# Long Weekends, Holidays, Festive seasons do make postive impact on ROI. 
# Compare to normal days movies gets better ROI on Festive seasons, holiday & Longweekends.


# In[221]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[222]:


plt.hist(movie_df['Budget'])


# In[223]:


sn.distplot(movie_df['Budget'],color='red')


# In[224]:


# Observations - 
# Most movies falls under low budget bracket


# In[225]:


sn.boxplot(x='Genre',y='YoutubeLikes',data=movie_df)


# In[213]:


# Observation
# Average likes for action genre is more than anyother genre. 


# In[242]:


movie_df_month.info()


# In[266]:


ROI_as_per_genre = movie_df_month.groupby('Genre')['Genre','ROI'].mean()
ROI_as_per_genre


# In[267]:


# Observations - 
# 'Drame' genre has gained highest ROI any other genre. 


# In[ ]:





# In[ ]:




