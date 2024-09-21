#!/usr/bin/env python
# coding: utf-8

# In[217]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Binomial Distribution

# In[218]:


from scipy import stats


# In[219]:


# Expected number of successful trial - 5
# Total number of trial - 20
# Success probability - 0.1

stats.binom.pmf(5,20,0.1)


# In[220]:


# range(0,21) - Will choose all the values from 0 to 20 except 21


# In[221]:


pmf_df=pd.DataFrame({'success':range(0,21),'pmf':list(stats.binom.pmf(range(0,21),20,0.1))})


# In[222]:


sn.barplot(x=pmf_df.success,y=pmf_df.pmf)
plt.xlabel('no of successful trail')
plt.ylabel('pmf')


# In[223]:


stats.binom.cdf(5,20,0.1)


# In[224]:


1-stats.binom.cdf(5,20,0.1)


# In[225]:


mean,var = stats.binom.stats(20,0.1)
print('average:',mean,'Variance:',var)


# # Poisson Distribution

# In[226]:


stats.poisson.cdf(5,10)


# In[227]:


1-stats.poisson.cdf(30,30)


# In[228]:


pmf_df=pd.DataFrame({'success':range(0,30),'pmf':list(stats.poisson.pmf(range(0,30),10))})


# In[229]:


sn.barplot(x=pmf_df.success,y=pmf_df.pmf)
plt.xlabel('Number of call received')


# # Normal Distribution

# In[230]:


beml_df=pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files)-2/Data/BEML.csv')
beml_df


# In[231]:


glaxo_df=pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files)-2/Data/GLAXO.csv')
glaxo_df


# In[232]:


beml_df=beml_df[['Date','Close']]
beml_df.head(10)


# In[233]:


glaxo_df=glaxo_df[['Date','Close']]
glaxo_df.head(10)


# In[234]:


glaxo_df=glaxo_df.set_index(pd.DatetimeIndex(glaxo_df['Date']))
beml_df=beml_df.set_index(pd.DatetimeIndex(beml_df['Date']))


# In[235]:


glaxo_df.head(10)


# In[236]:


beml_df.head(10)


# In[237]:


plt.plot(glaxo_df.Close)
plt.xlabel('Time')
plt.ylabel('Close Price')


# In[238]:


plt.plot(beml_df.Close)
plt.xlabel('Time')
plt.ylabel('Close Price')


# In[239]:


glaxo_df['gain']=glaxo_df.Close.pct_change(periods=1)
beml_df['gain']=beml_df.Close.pct_change(periods=1)


# In[240]:


glaxo_df[0:10]


# In[241]:


glaxo_df.dropna()


# In[242]:


beml_df[0:10]


# In[243]:


beml_df.dropna()


# In[244]:


plt.plot(glaxo_df.index,glaxo_df.gain)
plt.figsize=(8,6)
plt.xlabel('Time')
plt.ylabel('gain')


# In[245]:


plt.plot(beml_df.index,beml_df.gain)
plt.xlabel('Time')
plt.ylabel('gain')


# In[246]:


sn.distplot(glaxo_df.gain,label='Glaxo')
sn.distplot(beml_df.gain,label='Beml')
plt.xlabel('gain')
plt.ylabel('Density')
plt.legend()
plt.figsize=(10,8)


# In[247]:


beml_df.gain.describe()


# # Confidential Interval

# In[248]:


from scipy import stats
glaxo_df_ci=stats.norm.interval(0.95,loc=beml_df.gain.mean(),scale=beml_df.gain.std())


# In[249]:


print('gain at 95% confidence interval is :', np.round(glaxo_df_ci,4))


# In[250]:


beml_df_ci=stats.norm.interval(0.95,loc=beml_df.gain.mean(),scale=beml_df.gain.std())


# In[251]:


print('gain at 95% confidence interval is :', np.round(beml_df_ci,4))


# In[252]:


# Probability of loss 2% or higher in glaxo

print('Probability of making 2% loss or higher in glaxo:')
stats.norm.cdf(-0.02,loc=glaxo_df.gain.mean(),scale=glaxo_df.gain.std())


# In[253]:


# Probability of loss higher than 2% or loss in BEML

print('Probability of making 2% loss or higher in beml:')
stats.norm.cdf(-0.02,loc=beml_df.gain.mean(),scale=beml_df.gain.std())


# # Hypothesis Testing

# In[254]:


passport_df=pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files) - U Dinesh/Data/passport.csv')


# In[255]:


passport_df1 = passport_df.head()


# In[256]:


average = np.mean(passport_df1['processing_time'])
average


# In[257]:


print(list(passport_df.processing_time))


# # Z test

# In[258]:


import math


# In[259]:


def z_test(pop_mean,pop_std,sample):
    z_score=(sample.mean()-pop_mean)/(pop_std/math.sqrt(len(sample)))
    return z_score,stats.norm.cdf(z_score)


# In[260]:


z_test(30,12.5,passport_df.processing_time)


# In[261]:


# First value is Z value & Second value is p value (Probability value)
# As p values is more than 0.05 which is significance value null hypothesis is correct also 
# z value is more than -1.64


# # One sample T- test

# In[262]:


# one sample t test is used when Standard deviation is unknown


# In[263]:


bollywoodmovies_df =pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files) - U Dinesh/Data/bollywoodmovies.csv')


# In[264]:


bollywoodmovies_df.head()


# In[265]:


print(list(bollywoodmovies_df.production_cost))


# In[266]:


stats.ttest_1samp(bollywoodmovies_df.production_cost,500)


# In[267]:


# t-statstics is -2.284 and p values is 0.027, as p value is less than 0.05, 
# we can conculde that sample mean rejects production cost equal to 500


# # Two sample T test

# In[268]:


# when two population mean is to be tested & standard deviation is unknown


# In[269]:


healthdrink__df=pd.read_excel('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files) - U Dinesh/Data/healthdrink.xlsx','healthdrink_yes')


# In[270]:


healthdrink_yes_df.head(5)


# In[271]:


healthdrink__df=pd.read_excel('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files) - U Dinesh/Data/healthdrink.xlsx','healthdrink_no')


# In[272]:


healthdrink_no_df.tail()


# In[273]:


sn.distplot( healthdrink_yes_df['height_increase'], label ='healthdrink_yes' )
sn.distplot( healthdrink_no_df['height_increase'], label ='healthdrink_no' )
plt.legend();


# In[274]:


stats.ttest_ind(healthdrink_yes_df['height_increase'],healthdrink_no_df['height_increase'])


# # ANOVA - Analysis of variance

# In[285]:


onestop_df=pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files) - U Dinesh/Data/onestop.csv')


# In[286]:


onestop_df.head(5)


# In[287]:


sn.distplot(onestop_df['discount_0'],label='No discount')
sn.distplot(onestop_df['discount_10'],label='10% discount')
sn.distplot(onestop_df['discount_20'],label='20% discount')
plt.legend()


# In[288]:


from scipy.stats import f_oneway


# In[289]:


f_oneway(onestop_df['discount_0'],
        onestop_df['discount_10'],
        onestop_df['discount_20'])


# In[291]:


# P value is more less than 0.05 which means that sales quntity values under different discounts are different


# In[ ]:





# In[ ]:




