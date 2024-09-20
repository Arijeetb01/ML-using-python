#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np


# In[114]:


# Loading Dataset


# In[115]:


ipl_auction_df= pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files)/Data/IPL IMB381IPL2013.csv')


# In[116]:


ipl_auction_df


# In[117]:


type(ipl_auction_df)


# In[118]:


ipl_auction_df.head(10)


# In[119]:


ipl_auction_df.tail(10)


# In[120]:


list(ipl_auction_df)


# In[121]:


ipl_auction_df.head(10).transpose()


# In[122]:


ipl_auction_df.shape


# In[123]:


ipl_auction_df.info()


# In[124]:


ipl_auction_df[0:5]


# In[125]:


ipl_auction_df[-5:]


# In[126]:


ipl_auction_df['PLAYER NAME'][0:5].reset_index()


# In[127]:


ipl_auction_df[['PLAYER NAME','COUNTRY']][0:10]


# In[128]:


ipl_auction_df[['PLAYER NAME','COUNTRY','PLAYING ROLE']][0:10]


# In[129]:


ipl_auction_df.iloc[4:9,1:4]


# In[130]:


ipl_auction_df.COUNTRY.value_counts()


# In[131]:


ipl_auction_df.TEAM.value_counts()


# In[132]:


ipl_auction_df.COUNTRY.value_counts(normalize=True)*100


# In[133]:


pd.crosstab(ipl_auction_df['AGE'],ipl_auction_df['PLAYING ROLE'])


# In[134]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']]


# In[135]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE')[0:10]


# In[136]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE',ascending=False)[0:10]


# In[137]:


ipl_auction_df['PREMIUM']=ipl_auction_df['SOLD PRICE']-ipl_auction_df['BASE PRICE']


# In[138]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE','BASE PRICE','PREMIUM']]


# In[139]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE','BASE PRICE','PREMIUM']].sort_values('PREMIUM',ascending=False)


# In[140]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE','BASE PRICE','PREMIUM']].sort_values('PREMIUM',ascending=False)[0:10]


# In[141]:


soldprice_by_age=ipl_auction_df.groupby('AGE')['SOLD PRICE'].mean().reset_index()
soldprice_by_age


# In[142]:


soldprice_by_age_role=ipl_auction_df.groupby(['AGE','PLAYING ROLE'])['SOLD PRICE'].mean().reset_index()
soldprice_by_age_role


# In[143]:


soldprice_comparision=soldprice_by_age_role.merge(soldprice_by_age,on='AGE',how='outer')
soldprice_comparision


# In[144]:


soldprice_comparision.rename(columns={'SOLD PRICE_x':'SOLD_PRICE_AGE_ROLE','SOLD PRICE_y':'SOLD_PRICE_AGE'},inplace=True)
soldprice_comparision.head(10)


# In[145]:


ipl_auction_df[ipl_auction_df['SIXERS']>80][['PLAYER NAME','SIXERS']]


# In[146]:


ipl_auction_df[ipl_auction_df['AVE-BL']<50][['PLAYER NAME','AVE-BL']].sort_values('AVE-BL',ascending=False)[0:10]


# In[147]:


ipl_auction_df.drop('Sl.NO.',inplace=True,axis=1)


# In[148]:


ipl_auction_df


# In[149]:


ipl_auction_df


# In[150]:


#Handling missing Values

#SSV File - Space separted Values


# In[151]:


autos=pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files)/Data/auto-mpg.data',sep='\s+',header=None)


# In[152]:


autos.head(10)


# In[153]:


type(autos)


# In[154]:


autos.columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','year','origin','name']
autos.head(5)


# In[155]:


autos.info()


# In[156]:


autos['horsepower']=pd.to_numeric(autos['horsepower'],errors='coerce')
autos.info()


# In[157]:


autos[autos.horsepower.isnull()]


# In[158]:


autos1=autos.dropna(subset=['horsepower'])


# In[159]:


autos1[autos.horsepower.isnull()]


# # Data Visualization

# In[187]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[161]:


# Bar chart


# In[188]:


soldprice_by_age


# In[189]:


sn.barplot(x='AGE',y='SOLD PRICE',data=soldprice_by_age)


# In[164]:


soldprice_comparision


# In[193]:


sn.barplot(x='AGE',y='SOLD_PRICE_AGE_ROLE',hue='PLAYING ROLE',data=soldprice_comparision)


# In[166]:


# Histogram


# In[196]:


plt.hist([ipl_auction_df['SOLD PRICE']],bins=20)


# In[168]:


# Density or distribution Plot


# In[199]:


sn.distplot(ipl_auction_df['SOLD PRICE'])


# In[170]:


# Scatter Plot


# In[205]:


ipl_batsman_df=ipl_auction_df[ipl_auction_df['PLAYING ROLE']=='Batsman']
plt.scatter(x=ipl_batsman_df['SIXERS'],y=ipl_batsman_df['SOLD PRICE'])


# In[206]:


ipl_allrounder_df=ipl_auction_df[ipl_auction_df['PLAYING ROLE']=='Allrounder']
plt.scatter(x=ipl_allrounder_df['SIXERS'],y=ipl_allrounder_df['SOLD PRICE'])


# In[207]:


ipl_bowler_df=ipl_auction_df[ipl_auction_df['PLAYING ROLE']=='Bowler']
plt.scatter(x=ipl_bowler_df['SIXERS'],y=ipl_bowler_df['SOLD PRICE'])


# In[174]:


# To draw direction of relationship


# In[209]:


sn.regplot(x=ipl_batsman_df['SIXERS'],y=ipl_batsman_df['SOLD PRICE'])


# In[210]:


sn.regplot(x=ipl_allrounder_df['SIXERS'],y=ipl_allrounder_df['SOLD PRICE'])


# In[211]:


sn.regplot(x=ipl_bowler_df['SIXERS'],y=ipl_bowler_df['SOLD PRICE'])


# In[178]:


# Pair Plot


# In[213]:


influential_features=['SR-B','AVE','SIXERS','SOLD PRICE']
sn.pairplot(ipl_auction_df[influential_features],size=2)


# In[180]:


autos.info()


# In[216]:


influential_features1=['mpg','cylinders','displacement','horsepower']
sn.pairplot(autos[influential_features1],size=2)


# In[182]:


# Heatmap


# In[217]:


ipl_auction_df[influential_features].corr()


# In[219]:


sn.heatmap(ipl_auction_df[influential_features].corr(),annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
