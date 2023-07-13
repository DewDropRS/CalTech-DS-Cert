#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
# 
# A banking institution requires actionable insights from the perspective of Mortgage-Backed Securities, Geographic Business Investment and Real Estate Analysis. 
# 
# The objective is to identify white spaces/potential business in the mortgage loan. The mortgage bank would like to identify potential monthly mortgage expenses for each of region based on factors which are primarily monthly family income in a region and rented value of the real estate. Some of the regions are growing rapidly and Competitor banks are selling mortgage loans to subprime customers at a lower interest rate. The bank is strategizing for better market penetration and targeting new customers. A statistical model needs to be created to predict the potential demand in dollars amount of loan for each of the region in the USA. Also, there is a need to create a dashboard which would refresh periodically post data retrieval from the agencies. This would help to monitor the key metrics and trends.
# 

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 10)
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# 1.	Import data 

# In[52]:


# 
from google.colab import files
# 
uploaded = files.upload()


# In[53]:


df=pd.read_csv("train.csv")
df.head()


# In[54]:


# Check dimensionality of your data frame using shape
# This data frame has 27,321 observations or rows and 80 variables or columns 
df.shape


# In[55]:


# Check for duplicates
dups = df.duplicated(keep='first')
print(dups.value_counts())
print(dups[dups == True])


# In[56]:


# There are 160 duplicate rows
# Keep first record of all duplicated rows
df.drop_duplicates(inplace=True)


# In[57]:


# Recheck the dimensionality of the data
df.shape


#     2.	Figure out the primary key and look for the requirement of indexing

# In[58]:


# Check varialbes for unique values
df.nunique()


# In[59]:


# After removing duplicate records, I can now use UID as my primary key for indexing
# set UID as my index
df.set_index('UID',inplace = True)
df.head()


# In[ ]:


df.info()


#     3.	Gauge the fill rate of the variables and devise plans for missing value treatment. Please explain explicitly the reason for the treatment chosen for each variable. 

# In[61]:


df.isnull().sum().sum()


# In[62]:


df.isnull().sum()


# In[63]:


# Drop fields
# drop BLOCKID since it has many nulls
# drop SUMLEVEL since it is populated with a constant for all rows
df.drop(columns=['BLOCKID', 'SUMLEVEL'], inplace = True)


# In[64]:


df.isnull().sum()


# In[65]:


# either replace numeric values with mean or drop records if small amount of missing values
df.dropna(inplace = True)
df.shape


# In[66]:


df.isnull().sum().sum()


# In[67]:


# 576 Records that contained NAs dropped. This accounts to 2.1% of the data. This is OK.


# ### Exploratory Analysis

#   4.	Understanding homeowner costs are incredibly valuable because it is positively correlated to consumer spending which drives the economy through disposable income. Perform debt analysis. You may want to follow the following steps:
#   (my note:we want to explore locations where debt is not too high because that would mean less consumer spending;less homeowner costs means more consumer spending)

# *  (a) Explore the top 2,500 locations where the percentage of households with a second mortgage is the highest and percent ownership is above 10%. Visualize using geo-map. You may keep the upper limit for the percent of households with a second mortgage to roughly 50%.
# 

# In[68]:


df


# In[69]:


df2 = df[['place','pct_own','second_mortgage','lat','lng']][(df['pct_own']>0.1)&(df['second_mortgage']<=0.5)]


# In[70]:


df3_top_2500_locations=df2.sort_values(by='second_mortgage', ascending = False).head(2500)


# In[71]:


df3_top_2500_locations


# In[72]:


import plotly.graph_objects as go


# In[73]:


#export for Tableau geo map
df3_top_2500_locations.to_excel("df3_top_2500_locations.xlsx",sheet_name='top_2500')  


# In[74]:


fig = go.Figure(data = go.Scattergeo(lat=df3_top_2500_locations['lat'], lon=df3_top_2500_locations['lng'], text=df3_top_2500_locations['place']), 
                layout = dict(title = 'Top 2,500 locations with the highest second mortgages', geo = dict(scope = 'usa')))


# In[75]:


fig


# *  4 (b) Use the following bad debt equation: Bad Debt = P (Second Mortgage ∩ Home Equity Loan) 
# *    Bad Debt = second_mortgage + home_equity - home_equity_second_mortgage 

# In[76]:


# home_equity_second_mortgage-percent of houses with a second mortgage and home equity loan
# in order to not overcount this intersection needs to be subtracted from (second_mortgage + home_equity)
df['bad_debt']=df['second_mortgage'] + df['home_equity'] - df['home_equity_second_mortgage']
df['good_debt']=df['debt']-df['bad_debt']
df['remaining_income']=df['family_mean']-df['hi_mean']


# * 4 (c) Create pie charts to show overall debt and bad debt

# In[77]:


df_pie = [df['good_debt'].sum(),df['bad_debt'].sum()]
df_pie


# In[78]:


plt.pie(df_pie , labels = ['Good Debt', 'Bad Debt'], autopct='%0.1f%%');


# *  4 (d) Create Box and whisker plot and analyze the distribution for 2nd mortgage, home equity, good debt, and bad debt for different cities

# In[79]:


import random


# In[80]:


#Find the top 5 most occuring records by city
df['city'].value_counts().head(5)


# In[81]:


#all_cities = [df['city'].unique()]
#run below once
#print(np.random.choice(all_cities[0], size=5, replace=False))


# In[82]:


cities = ['Chicago', 'Brooklyn', 'Los Angeles', 'Houston' ,'Philadelphia']


# In[83]:


df_cities = df.loc[df['city'].isin(cities)]
df_cities


# In[84]:


import seaborn as sns
fig , axes = plt.subplots(ncols=2,nrows=2, figsize=(20, 20))
sns.boxplot(y='second_mortgage', x='city', data=df_cities, ax=axes[0,0])
sns.boxplot(y='home_equity', x='city', data=df_cities, ax=axes[0,1])
sns.boxplot(y='good_debt', x='city', data=df_cities, ax=axes[1,0])
sns.boxplot(y='bad_debt', x='city', data=df_cities, ax=axes[1,1])


# ### Findings from boxplots:
# Los Angeles and Chicago have the highest median perentage of second mortgages from the selected cities.LA also has the highest median of percentage of households with home equity loans. LA and Chicago lead in good debt and bad debt.

# e) Create a collated income distribution chart for family income, house hold income, and remaining income

# In[85]:


fig, axes=plt.subplots(ncols=3, nrows=1, figsize=(15,5))
# family income - Multiple incomes under one family.
sns.histplot(x=df['family_mean'], ax=axes[0])
# household income - all income under one address
sns.histplot(x=df['hi_mean'], ax=axes[1])
# remaining_income = family_mean - hi_mean
sns.histplot(x=df['remaining_income'], ax=axes[2])


# Family mean income is mostly distributed between 30K-100K. 
# Household income is mostly distributed between 25k-125K.

# 1. Perform EDA and come out with insights into population density and age. You may have to derive new fields (make sure to weight averages for accurate measurements):
# 
# a) Use pop and ALand variables to create a new field called population density
# 
# b) Use male_age_median, female_age_median, male_pop, and female_pop to create a new field called median age c) Visualize the findings using appropriate chart type

# In[34]:


df['population_density']=df['pop']/df['ALand']


# In[35]:


df['median_age'] = ((df['male_pop']*df['male_age_median'])+(df['female_pop']*df['female_age_median']))\
/(df['male_pop']+df['female_pop'])


# In[36]:


fig, axes=plt.subplots(ncols=1, nrows=1, figsize=(10,10))
sns.histplot(x=df['median_age'])


# Median age distribution mostly between 30-50 years.

# 2. Create bins for population into a new variable by selecting appropriate class interval so that the number of categories don’t exceed 5 for the ease of analysis.
# 
#   a) Analyze the married, separated, and divorced population for these population brackets
# 
#   b) Visualize using appropriate chart type

# In[37]:


df['pop'].describe()


# In[38]:


#figuring out cut
53812/3


# In[39]:


# create a new column for population band
df['pop_bins']=pd.cut(df['pop'],bins=[0,18000,36000,54000])
df['pop_bins']


# In[40]:


chart = df.groupby(by ='pop_bins')[['married','separated','divorced']].mean()
chart


# In[41]:


chart.plot(kind='bar')


# This chart show me the percentages of marriage status by population bins. For example in each of the population categories, married makes up most of the proportion of the populations compared to separated and divorced.

# 3. Please detail your observations for rent as a percentage of income at an overall level, and for different states.
# 
# 

# In[42]:


#rent mean must be a monthly mean so multiply by 12 to get a full year's worth for calculating rent as a percentage of income
df['rent_mean'].mean()


# In[43]:


# Overall percentage of income

(df['rent_mean'].sum()*12)/df['family_mean'].sum()*100


# The overall rent spent as a percentage of family income is 16%

# In[44]:


# View this percentage by state
rent_percentage_by_state = ((df.groupby(by ='state')['rent_mean'].sum()*12)/(df.groupby(by ='state')['family_mean'].sum()))*100
rent_percentage_by_state.sort_values(ascending = False).head(5)


# The top 5 states that have the highest perentage of rent to family income are:
# Hawaii, California, Florida, Puerto Rico, and Nevada

# In[45]:


rent_percentage_by_state.sort_values().head(5)


# The top 5 states with the lowest rent to family income ratio are:
# North Dakota, South Dakota, Iowa, West Virginia, and Montana

# 4. Perform correlation analysis for all the relevant variables by creating a heatmap. Describe your findings.

# In[49]:


correlation_df = df[['family_mean', 'married', 'divorced', 'home_equity', 'hs_degree','median_age', 'second_mortgage', 'pct_own', 'bad_debt']]
a = correlation_df.corr()


# In[50]:


a.to_excel("correlation.xlsx",sheet_name='correlation')


# In[145]:


sns.heatmap(a)


# ### Findings from correlation heatmap:
# Family mean income is positively correlated to percentage of people with at least a high school degree. Married people is positively correlated to percentage of home ownership.

# In[51]:


df.to_excel("df.xlsx",sheet_name='df')


# Data Pre-processing:
# 
# 1. The economic multivariate data has a significant number of measured variables. **The goal is to find where the measured variables depend on a number of smaller unobserved common factors or latent variables.** 
# 2. Each variable is assumed to be dependent upon a linear combination of the common factors, and **the coefficients are known as loadings**. Each measured variable also includes a component due to independent random variability, known as **“specific variance”** because it is specific to one variable. **Obtain the common factors and then plot the loadings.** **Use factor analysis** to find latent variables in our dataset and gain insight into the linear relationships in the data. Following are the list of latent variables:
# 
# • Highschool graduation rates
# 
# • Median population age
# 
# • Second mortgage statistics
# 
# • Percent own
# 
# • Bad debt expense

# In[56]:


# save the latent variables for factor analysis into a new dataframe
print(list(df.columns))


# In[133]:


df_fa = df[['family_mean', 'married', 'divorced', 'home_equity', 'hs_degree','median_age', 'second_mortgage', 'pct_own', 'bad_debt' ]]


# In[ ]:


df_fa


# In[ ]:


df_fa.isnull().sum().sum()


# In[135]:


#Standardize the features- mean 0, std 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df_fa = scaler.fit_transform(df_fa)


# In[136]:


scaled_df_fa


# In[137]:


# Attempting to reduce 10 features down to 3 using factor analysis
from sklearn.decomposition import FactorAnalysis 
#fa = FactorAnalysis(n_components = 3, rotation = 'varimax', max_iter = 500)
# modified by increasing factors to 4 to achieve a better R2 score
fa = FactorAnalysis(n_components = 4, rotation = 'varimax', max_iter = 500)
X_transformed = fa.fit_transform(scaled_df_fa)
pd.DataFrame(X_transformed)


# In[138]:


#loadings = pd.DataFrame(fa.components_, index = ["Factor1", "Factor2", "Factor3"], columns = df_fa.columns).T
loadings = pd.DataFrame(fa.components_, index = ["Factor1", "Factor2", "Factor3", "Factor4"], columns = df_fa.columns).T
loadings


# Factor Analysis helps us find the latent variables
# -  Factor1: (home_equity, second_mortgage, bad debt) They are all related and we can think of this factor as being **Bad Payers**.
# -  Factor2:(married, median_age, pct_own) **Owners**
# -  Factor3: **Divorced**
# -  Factor4:(family_mean, hs_degree). **Income/Education**

# - positive coefficient: independent variable increases -> the mean of the dependent variable increases. 
# - negative coefficient: independent variable increases -> the dependent variable decreases.

# In[139]:


plt.figure(figsize =(15,5))
plt.plot(loadings)
plt.legend(['F1-Bad Payers', 'F2-Owners', 'F3-Divorced', 'F4-Familyl Income/Education'])


# In[140]:


# Prepare the dataframe for model
# Convert array to a dataframe
factors_df = pd.DataFrame(X_transformed)
# Target
label = pd.DataFrame(df.hc_mortgage_mean)
label.reset_index(inplace = True, drop = True)
#combine into one dataframe
final_df = pd.concat([factors_df, label], axis = 1)
final_df


# # Data Modeling :
# 1. Build a linear Regression model to predict the total monthly expenditure for home mortgages loan. Please refer ‘deplotment_RE.xlsx’. Column hc_mortgage_mean is predicted variable. This is the mean monthly     mortgage and owner costs of specified geographical location. Note: Exclude loans from prediction model which have NaN (Not a Number) values for hc_mortgage_mean.
# 
# -  a) Run a model at a Nation level. If the accuracy levels and R square are not satisfactory proceed to below step.
# 
#   b) Run another model at State level. There are 52 states in USA.
# 
#   c) Keep below considerations while building a linear regression model.
# > Data Modeling:
# > -  variables should have significant impact on predicting Monthly mortgage and owner costs
# > - Utilize all predictor variable to start with initial hypothesis
# > - R square of 60 percent and above should be achieved
# > - Ensure Multi-collinearity does not exist in dependent variables
# > - Test if predicted variable is normally distributed

# In[141]:


# Check there are no missing values for the target variable
final_df['hc_mortgage_mean'].isnull().sum().sum()


# In[142]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = final_df.drop(columns = 'hc_mortgage_mean')
y = final_df.hc_mortgage_mean # this is a series
lm.fit(X,y)
y_predict = lm.predict(X)


# In[143]:


round(lm.score(X,y)*100,1)


# Success. I was able to come up with a model whose R squared score is above 60%.

# In[130]:


y_predict


# In[131]:


#Test if predicted variable is normally distributed
sns.histplot(y_predict, kde = True)


# The plot has a bell shaped curve and so is normally distributed.

# In[ ]:


# left off at ~ 1:21:00
# Next part is to create the Tableau dashboard.
# make some more remarks here for all of your charts that are missing findings

