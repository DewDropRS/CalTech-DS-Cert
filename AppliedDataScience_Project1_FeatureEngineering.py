#!/usr/bin/env python
# coding: utf-8

# ## Project 1 - Feature Engineering
# ### CB DS JUN 2022 COHORT - Rocio Segura
# 
# 

# In[1]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[2]:


# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', 50)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Download the “PEP1.csv” using the link given in the Feature Engineering project problem statement

# In[3]:


# on MAC, right click on file in finder, then click option, the select Copy "filename.csv" as Pathname
# /Users/digitalworkstation/Desktop/CalTech Data Science Certification Program/Applied_Data_Science_Python/Project 1/PEP1.csv
# If you need to, load your data to a csv file:
# df.to_csv("")
# for excel, use read_excel & to_excel
df=pd.read_csv("PEP1.csv")
#new_na_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', '#N/A', 'N/A', 'n/a', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
#df = pd.read_csv("/Users/digitalworkstation/Desktop/CalTech Data Science Certification Program/Applied_Data_Science_Python/Project 1/Project_Dataset_Feature_Engineering/PEP1.csv", keep_default_na=False, na_values=new_na_values)
#df = pd.read_csv("/content/drive/MyDrive/CalTech Data Science Certification Program/Applied_Data_Science_Python/Project 1/Project_Dataset_Feature_Engineering/PEP1.csv")


# ## 1 Understand the dataset
# - a. Identify the shape of the dataset
# - b. Identify variables with null values
# - c. Identify variables with unique values
# 

# ## 1.a. Identify the shape of the dataset

# In[4]:


# Check dimensionality of your data frame using shape
# This data frame has 1460 observations or rows and 81 variables or columns 
df.shape


# In[5]:


# confirm the type of data frame
# to view data type of a specific column, df['colname'].dtype
type(df)


# In[6]:


# info()
# outputs range index and the following:
#  #, Column, Non-Null, Count, Dtype
df.info()


# In[7]:


# df[''].unique() - returns numpy array with all unique values of a column
# df[''].values() - returns numpy array with all the values of a column
# df.mean()


# In[8]:


# Mode - locate the central tendency of the numeric or nominal data (data that can be labeled or classified into mutually exclusive categories).
# df.mode(axis=0)  (0 or ‘index’ : get mode of each column, 1 or ‘columns’ : get mode of each row)
# Median-separates upper half of data from the lower half
# df.median()


# In[9]:


df.describe().transpose()


# ## 1.b. Identify variables with null values

# ### First I want to know if we have data for entire years.

# In[10]:


df.groupby('YrSold')['MoSold'].value_counts().unstack()


# ### It looks like 2010 only has data for Jan - July. So I will drop the data for YrSold in 2010

# In[11]:


# Permanently drop all rows from the dataframe where the year sold is 2010.
df.drop(df[df['YrSold'] == 2010].index, inplace = True)
# What is the shape of my data frame now?
df.shape


# In[12]:


# Here I'm choosing to use isna() and printing only those features
# that have one or more null values
naseries = df.isna().sum().sort_values(ascending = False)
# is there a better way of doing the following?
print(naseries[naseries > 0])


# In[13]:


# As part of this exercise, I will drop features that have more than 50% of data missing since they will not be useful in a model.
cols2drop = df.columns[df.isnull().mean()>0.5]
df.drop(cols2drop, axis=1, inplace = True)
print(cols2drop)


# In[14]:


#How many features am I left with and how many rows do I have?
# 1285 rows and 77 features
df.shape


# In[15]:


# 1.c. Identify variables with unique values
# There is only one variable that has 1285 unique values and that is ID
# I can use this later on to identify categorical features that have too many unique values to work with and that will need some type of recoding of categories like having an "other" category.
df.nunique().sort_values(ascending = False)


# In[16]:


#Example of how to fill in null values but I have dropped Alley. 
#I may use this further down.
#df['Alley'].fillna('Does not apply', inplace=True)
#df['Alley']


# ## 2 Generate a separate dataset for numerical and categorical variables

# ### Numerical subset 

# In[17]:


# numerical subset of df
df_numeric = df.loc[:,df.dtypes != np.object]
df_numeric.info()


# In[18]:


# numerical subset of df cont.
df_numeric.head()


# ### There are clearly some features that should be considered ordinal categorical: 
#      MSSubClass, OverallQual, OverallCond, GarageYrBlt, YearBuilt, YearRemodAdd, MoSold, YrSold

# In[19]:


#first remove these features from df_numeric
df_numeric.drop(columns=['MSSubClass', 'OverallQual', 'OverallCond', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold'], inplace = True)
df_numeric.head()


# In[20]:


#What is the shape of my numeric subset?
df_numeric.shape


# ### Categorical subset

# In[21]:


#create a new dataframe that contains the additional category columns (the ones dropped from the numeric subset)
additional_cat_cols = df.filter(['MSSubClass', 'OverallQual', 'OverallCond', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold'], axis = 1)
additional_cat_cols.head()


# In[22]:


# categorical - concatenating objects and additional categorical features
df_object = df.loc[:,df.dtypes == np.object]
df_categorical = pd.concat([df_object, additional_cat_cols, df['SalePrice']], axis=1)
# add SalePrice


# In[23]:


df_categorical.info()


# In[24]:


# Let's take a look at our categorical features
df_categorical.head()


# ## 3 EDA of numerical variables
# -    a. Missing value treatment
# -    b. Identify the skewness and distribution
# -    c. Identify significant variables using a correlation matrix 
# -    d. Pair plot for distribution and density
# 

# ### 3.a. Missing value treatment

# In[25]:


# What are my missing values for numeric data set?
naseries_num = df_numeric.isna().sum()
print(naseries_num[naseries_num > 0])


# In[26]:


# LotFrontage: Linear feet of street connected to property-I will fill missing values for this feature.
# MasVnrArea: Masonry veneer area in square feet- This feature has very few missing that I will ignore filling null values.


# In[27]:


# I first want to see the distribution of LotFrontage to decide which method I will use for filling missing values.
plt.figure(figsize=(10,10))
sns.distplot(df_numeric['LotFrontage'])


# In[28]:


# Is the median of LotFrontage the same as mean? 
df_numeric['LotFrontage'].agg(['mean', 'median', 'min', 'max'])


# In[29]:


# Here I will fill LotFrontage with the median. Mean and Median are very close but outlier
# is causing it to skew slightly.
df_numeric.LotFrontage.fillna(69.0, inplace = True)


# ### In this step, I will examine outliers using box plots. Then I will systematically remove outliers (values more than 3 std from mean).

# In[30]:


# I currently have 30 features in df_numeric
df_numeric.shape


# In[54]:


def createBoxplots(data):
    plt.tight_layout()
    plt.figure(figsize=(20,200))
    for i,j in enumerate(data.columns):
        # 10 rows, 3 columns
        plt.subplot(10,3,i+1)
        sns.boxplot(data=data, y=j)
    plt.show()
createBoxplots(df_numeric)


# In[32]:


#first make a deep copy of the data
df_numeric_test = df_numeric.copy(deep = True)
df_numeric_trimmed=df_numeric_test.mask(df_numeric_test.sub(df_numeric_test.mean()).div(df_numeric_test.std()).abs().gt(3))


# In[33]:


df_numeric_trimmed.describe().transpose()


# ### Correlations for numerical features with target SalePrice

# In[34]:


pd.set_option('display.max_rows', None)
df_numeric_trimmed.corr()['SalePrice'].abs().sort_values(ascending=False)


# In[35]:


# Why NaN for BsmtHalfBath, KitchebvGr, PoolArea. These features have 1 non-NaN value. I will drop these features.
#df_numeric_trimmed['BsmtHalfBath'].value_counts(normalize = True)
#df_numeric_trimmed['KitchebvGr'].value_counts(normalize = True)
df_numeric_trimmed['PoolArea'].value_counts(normalize = True)


# In[36]:


features2drop = df_numeric_trimmed.columns[df_numeric_trimmed.corr()['SalePrice'].abs() < 0.5]


# In[37]:


#Drop features with low correlation scores and drop ones that have NaN correlation score.
df_numeric_trimmed.drop(columns=features2drop, inplace = True)
df_numeric_trimmed.drop(columns=['BsmtHalfBath','KitchebvGr','PoolArea'], inplace = True)


# In[38]:


#Examine new shape of df_numeric_trimmed. I now have 11 features of interest
df_numeric_trimmed.shape


# In[39]:


df_numeric_trimmed.head()


# In[40]:


#Here I want to see their pair plots
sns.pairplot(df_numeric_trimmed)
plt.show()


# ## 4	EDA of categorical variables
# -  a. Missing value treatment
# -	b.	Count plot and box plot for bivariate analysis
# -	c.	Identify significant variables using p-values and Chi-Square values
# 	

# ### 4a Missing value treatment

# In[41]:


df_categorical.isna().sum().sort_values(ascending = False)
#Here I'm making the assumption that missing values for condition and quality features mean that the home does not have that feature.


# In[42]:


# Create a new dataframe for categorical features that are ordinal and can be recoded 0-5
df_5ScaleCats = df_categorical.filter(['ExterQual','ExterCond', 'BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC'], axis = 1)
df_5ScaleCats.info()


# In[43]:


charlist = ["Ex", "Gd", "TA", "Fa", "Po", "0"]
num_list = ["5", "4","3","2", "1", "0"]
char2Num = dict(zip(charlist,num_list))
char2Num


# In[44]:


# Not sure if recoding NaNs to 0 is the right move but did it anyway
df_5ScaleCats = df_5ScaleCats.fillna('0')


# In[45]:


df_5ScaleCats = df_5ScaleCats.applymap(lambda x: char2Num[x] )


# In[46]:


#I need to convert recoded features from character to numeric.
dict_columns_type = {'ExterQual': int, 'ExterCond': int, 'BsmtQual': int,'BsmtCond': int,'HeatingQC': int,'KitchenQual': int,'FireplaceQu': int,'GarageQual': int,'GarageCond': int}
df_5ScaleCats = df_5ScaleCats.astype(dict_columns_type)


# In[47]:


# Check that data type conversion worked
df_5ScaleCats.info()


# In[48]:


# Check that recoding of features worked
df_5ScaleCats.apply(pd.value_counts)


# In[49]:


# Drop original features and add the new recoded ones
df_categorical.drop(columns=['ExterQual','ExterCond', 'BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond'], inplace = True)
# Add recoded features back in
df_categorical2 = pd.concat([df_categorical, df_5ScaleCats], axis=1)


# ### 4c Identify significant variables using the corr method

# In[50]:


pd.set_option('display.max_rows', None)
df_categorical2.corr()['SalePrice'].abs().sort_values(ascending=False)


# ### Keep features that have a correlation coefficient >= 0.5

# In[51]:


df_categorical_3 = df_categorical2.filter(['OverallQual', 'ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu', 'YearBuilt', 'YearRemodAdd'], axis = 1)


# ### 4b Count plots (moved this step here to generate count plots for only those features I'm interested in.)

# In[52]:


# For loop to generate seaborn count plots
fig , ax = plt.subplots(4,2,figsize = (50,50))
for i , subplots in zip (df_categorical_3 , ax.flatten()):
  sns.countplot(df_categorical_3[i],ax = subplots)
plt.show()


# ### Combine all the significant categorical and numerical variables
# 

# In[53]:


df_final = pd.concat([df_numeric_trimmed, df_categorical_3], axis=1)
df_final.info()

