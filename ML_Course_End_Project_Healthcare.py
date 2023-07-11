#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# Cardiovascular diseases are the leading cause of death globally. 
# It is therefore necessary to identify the causes and develop a system to predict heart attacks in an effective manner. 

# ## Task to be performed

# 1. Preliminary analysis
#     1. Perform preliminary data inspection and report the findings on the structure of the data, missing values, duplicates, etc.
#     1. Based on these findings, remove duplicates (if any) and treat missing values using an appropriate strategy
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', 50)
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_excel("cep1_dataset.xlsx")
df.head()


# ### 1.A- preliminary data inspection

# In[3]:


df.shape


# In[4]:


# The dataframe has 303 rows and 14 features


# In[5]:


df.isnull().sum()


# In[6]:


# No missing values in the data


# In[7]:


# Check for duplicates
dups = df.duplicated(keep='first')
print(dups.value_counts())
print(dups[dups == True])


# In[8]:


# remove duplicate row
df=df[~dups]


# In[9]:


df.shape


# ## 2.	Prepare a report about the data explaining the distribution of the disease and the related factors using the steps listed below:
#     - a. Get a preliminary statistical summary of the data and explore the measures of central tendencies and spread of the data
#     - b. Identify the data variables which are categorical and describe and explore these variables using the appropriate tools, such as count plot 
#     - c. Study the occurrence of CVD across the Age category
#     - d. Study the composition of all patients with respect to the Sex category
#     - e. Study if one can detect heart attacks based on anomalies in the resting blood pressure (trestbps) of a patient
#     - f. Describe the relationship between cholesterol levels and a target variable
#     - g. State what relationship exists between peak exercising and the occurrence of a heart attack
#     - h. Check if thalassemia is a major cause of CVD
#     - i. List how the other factors determine the occurrence of CVD
#     - j. Use a pair plot to understand the relationship between all the given variables
# 

# ### 2.a. Get a preliminary statistical summary of the data and explore the measures of central tendencies and spread of the data

# In[10]:


df['target'].value_counts()


# In[11]:


df[['age','trestbps','chol','thalach','oldpeak']].describe().transpose()


# ### 2.b. Identify the data variables which are categorical and describe and explore these variables using the appropriate tools, such as count plot

# In[12]:


# Create a new temp dataframe for categorical columns
df_cat = df.filter(['sex', 'cp', 'exang', 'thal','Target'], axis = 1)


# In[13]:


df.info()


# In[14]:


# For loop to generate seaborn count plots for categorical features
fig , ax = plt.subplots(2,2,figsize = (50,50))
sns.set(font_scale = 1.5)
for i , subplots in zip (df_cat , ax.flatten()):
    
    sns.countplot(df_cat[i],ax = subplots)
plt.show()


# ### 2.c. Study the occurrence of Cardio Vascular Disease (CVD) across the Age category

# In[15]:


df.groupby(['target'])['age'].describe()


# In[16]:


age_target=df.groupby("target")["age"].value_counts(ascending=False,bins=4)
age_target


# In[17]:


df['age'].hist(by=df['target'],bins=40)


# In[18]:


# Scatter plot y=target, x=age
plt.scatter(df['age'], df['target'])
plt.xlabel("Age")
plt.ylabel("CVD (Target)")
plt.show()


# In[19]:


# Age does not seem to be a predictor of CVD
# Their mean age, although not the same, are close across the target


# In[20]:


# Drop age from data frame
# df.drop(['age'], axis=1, inplace=True)


# In[21]:


df.shape


# ### 2.d. Study the composition of all patients with respect to the Sex category

# In[22]:


#rename sex column
#1=Male;0=female
df.rename(columns={'sex': 'sex_male'}, inplace=True)


# In[23]:


df['sex_male'].value_counts()


# In[24]:


male_target=df.groupby("target")["sex_male"].value_counts(ascending=False, normalize=True)*100
male_target


# In[25]:


# There seems to be an imbalance of data collected by sex where men are represented about twice that of women
# I will attempt to balance the data using downsampling
df_majority = df[df.sex_male==1]
df_minority = df[df.sex_male==0]


# In[26]:


#down sample majority class (sex_male=1)
#use random_state = 123 so that you can reproduce results later
from sklearn.utils import resample
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=96,     # to match minority class
                                 random_state=123) # reproducible results


# In[27]:


#combine majority and minority data
df_downsampled = pd.concat([df_majority_downsampled, df_minority])


# In[28]:


df_downsampled['sex_male'].value_counts()


# In[29]:


# re-examine again using this new resampled data frame
male_target=df_downsampled.groupby("target")["sex_male"].value_counts(ascending=False, normalize=True)*100
male_target


# In[30]:


pd.crosstab(index=df_downsampled['sex_male'], columns=df_downsampled['target'])


# In[31]:


pd.crosstab(index=df_downsampled['sex_male'], columns=df_downsampled['target'],normalize=True,margins=True,margins_name='Total')*100


# In[32]:


# We can see more clearly how sex plays a role in CVD
# 58% of the patients have been flagged as having CVD
# Of that 58%, females makeup 37.5% while males makeup 21.4%
# This seams counter intuitive since I thought that men are at higher risk for CVD
# But I will opt to go with this balanced data


# ### 2.e. Study if one can detect heart attacks based on anomalies in the resting blood pressure (trestbps) of a patient

# In[33]:


# scatter plot

plt.scatter(df_downsampled['trestbps'], df_downsampled['target'])
plt.xlabel("Resting Blood Pressure on Admission")
plt.ylabel("CVD (Target)")
plt.show()


# In[34]:


df_downsampled.groupby(['target'])['trestbps'].describe()


# In[35]:


# Resting blood pressure analysis
# There does not seem to be a correlation of anomalies of resting blood pressure to CVD


# ### 2.f. Describe the relationship between cholesterol levels and a target variable

# In[36]:


plt.scatter(df_downsampled['chol'], df_downsampled['target'])
plt.xlabel("Serum cholesterol in mg/dl")
plt.ylabel("CVD (Target)")
plt.show()


# In[37]:


s = sns.jointplot(x='chol', y='target', data=df_downsampled, kind="hex");
s.ax_joint.grid(False);
s.ax_marg_y.grid(False);
s.fig.suptitle("Total Serumn Cholesterol mg/dl");


# In[38]:


#Test creating an age_band column
#age_group_data=df_downsampled.filter(['age'], axis = 1)
#print(age_group_data)
#age_group_data['age_band'] = pd.cut(age_group_data['age'], bins=[20, 30, 40, 50, 60, 70, float('Inf')], labels=['20-29','30-39','40-49', '50-59', '60-69', '70-79'], right = False)
#pd.set_option('display.max_rows', None)
#print(age_group_data)


# In[39]:


df_downsampled.groupby(['target'])['chol'].describe()


# In[40]:


#Apply tested code
df_downsampled['age_band']=pd.cut(df_downsampled['age'], bins=[20, 30, 40, 50, 60, 70, float('Inf')], labels=['20-29','30-39','40-49', '50-59', '60-69', '70-79'], right = False)


# In[41]:


df_downsampled.groupby(['target','age_band'])['chol'].describe()


# In[42]:


df_downsampled.head()


# In[43]:


# It does not seem cholesterol is a good predictor for CVD
# The means by target are close CVD:251 mg/dl vs non-CVD:253 mg/dl


# ### 2.g. State what relationship exists between peak exercising and the occurrence of a heart attack

# In[44]:


# thalach
s = sns.jointplot(x='thalach', y='target', data=df_downsampled, kind="hex");
s.ax_joint.grid(False);
s.ax_marg_y.grid(False);
s.fig.suptitle("Maximum heart rate achieved (thalach)");


# In[45]:


plt.scatter(df_downsampled['thalach'], df_downsampled['target'])
plt.xlabel("Maximum heart rate achieved (thalach)")
plt.ylabel("CVD (Target)")
plt.show()


# In[46]:


df_downsampled.groupby(['target'])['thalach'].describe()


# In[47]:


# There seems to be a relationship between the maximum heart rate achieved during exercise and whether a patient is flagged with CVD
# The charts and stats show us higher heart rates for patients flagged as having CVD


# ### 2.h. Check if thalassemia is a major cause of CVD

# In[48]:


#thal: 
# 1 = fixed defect (no blood flow in some part of the heart); 
# 2 = normal blood flow; 
# 3 = reversable defect (a blood flow is observed but it is not normal) 


# In[49]:


df_downsampled_dummies = pd.get_dummies(df_downsampled, columns = ['thal','cp', 'restecg'])
df_downsampled_dummies.head()


# In[50]:


df_downsampled_dummies.groupby(['target'])['thal_3'].value_counts()


# ###  2.i. List how the other factors determine the occurrence of CVD

# In[51]:


import seaborn as sns
plt.figure(figsize=(15,10))
sns.set(style = 'white', color_codes = True)
sns.set(font_scale = 1)
sns.heatmap(df_downsampled_dummies.corr(),annot=True)


# In[52]:


pd.set_option('display.max_rows', None)
df_downsampled_dummies.corr()['target'].sort_values(ascending=False)


# ###  Select colums for modeling that have correlation >= 0.2

# In[53]:


df_model_data = ["thal_2", "thalach", "slope", "cp_2", "cp_1", "target"]
df_model_data = df_downsampled_dummies.reindex(columns=df_model_data)
df_model_data.head()


# ### 2.j. Use a pair plot to understand the relationship between all the given variables

# In[54]:


# Not helpful for binary features. 
# But can see a relationship between thalch and target where higher heart rates achieved are more associated with CVD
# than lower heart rates.
sns.pairplot(df_model_data)
plt.show()


# ## 3.	Build a baseline model to predict the risk of a heart attack using a logistic regression and random forest and explore the results while using correlation analysis and logistic regression (leveraging standard error and p-values from statsmodels) for feature selection

# In[103]:


# Model building imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from sklearn import metrics

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[124]:


# Prepare data for model
X = df.drop('target',axis=1)
y = df['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# Check the shape of each
print('X_train: ',  X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


# In[125]:


LogReg = LogisticRegression()
model_lg=LogReg.fit(X_train, y_train)


# In[126]:


y_pred_lg = model_lg.predict(X_test)


# In[127]:


cf_matrix = confusion_matrix(y_test, y_pred_lg)


# In[128]:


# Visualize the confusion matrix
# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
# As percentages
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues') 

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[117]:


print('Accuracy Score for Logistic Regression: ',
      round(accuracy_score(y_test,y_pred_lg),5)*100,'%')


# In[118]:


print(classification_report(y_test, y_pred_lg)) 


# In[119]:


# ROC
from sklearn.metrics import roc_auc_score,roc_curve
y_prob = model_lg.predict_proba(X_test)[:,1]
false_pos_rate,true_pos_rate,threshold = roc_curve(y_test,y_prob)


# In[120]:


#Plot ROC Curve
plt.figure(figsize=(10,6))
plt.title('Receiver Operating Characterstic')
plt.plot(false_pos_rate,true_pos_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[121]:


# What is the area under the curve?
roc_auc_score(y_test,y_prob)


# ### Random Forest Classifier using downsampled data (gender balanced data)

# In[152]:


#Trying with downsampled dataframe without dummy variables and dropping age band
RFC = RandomForestClassifier()
X = df_downsampled.drop(['target', 'age_band'],axis=1)
#X = df_model_data.drop(['target'],axis=1)
y = df_downsampled['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[153]:


param_grid = {'n_estimators':[100,150,200,250],'max_depth':[4,6,8]}
tuning = GridSearchCV(estimator=RFC, param_grid=param_grid, scoring='r2')
model_rfc=tuning.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)
tuning.best_params_,tuning.best_score_


# In[154]:


print('classification_report:\n',classification_report(y_test, y_pred_rfc))
confusion_matrix(y_test, y_pred_rfc)


# ### Random Forest Classifier using original data

# In[99]:


RFC = RandomForestClassifier()
param_grid = {'n_estimators':[100,150,200,250],'max_depth':[4,6,8]}
GSCV = GridSearchCV(estimator=RFC, param_grid=param_grid, scoring='r2')

X = df.drop('target',axis=1)
y = df['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[100]:


model_rfc=GSCV.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)
GSCV.best_params_,GSCV.best_score_


# In[97]:


print('Classification Report:\n',classification_report(y_test, y_pred_rfc))
confusion_matrix(y_test, y_pred_rfc)


# In[ ]:


## Accuracy Score is better using the original data. Gender balancing the data was not a good idea. 

