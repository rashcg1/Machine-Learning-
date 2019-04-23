#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


d=pd.read_csv(r'D:\HousePredictions\train.csv',encoding='unicode_escape')
d.head()


# In[3]:


d.describe()


# In[4]:


d.info()


# In[5]:


d_missing=d.isna().sum()


# missing[d_missing>0].sort_values(ascending=False)

# In[6]:


d_missing[d_missing>0].sort_values(ascending=False)


# In[7]:


#keeping only columns which dont have na


# In[8]:


d=d.dropna(axis=1, how='any')
d.shape


# In[9]:


#removing Id field which doesnt have impact on house price.
#del d['Id']
d.head()


# In[10]:


#corelation matrix


# In[86]:


import seaborn as sns
import matplotlib.pyplot as plt
matrix = d.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(matrix, vmax=0.7, square=True)


# In[87]:


#selcting only features which are highly correlated
tcf=matrix['SalePrice'].sort_values(ascending=False)


# In[88]:


# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)
tcf = tcf[abs(tcf) >= 0.6]


# In[89]:


tcf = tcf[tcf.index != 'SalePrice']
tcf


# In[90]:


tcf.shape


# In[91]:


cols = tcf.index.values.tolist() + ['SalePrice']
sns.pairplot(d[cols], size=2.5)
plt.show()


# In[94]:


# Build the correlation matrix
matrix = d[cols].corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, vmax=1.0, square=True)


# In[146]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X=d1.loc[:,d1.columns!='SalePrice']
y = d1['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[147]:


y_pred = model.predict(X_test)

# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()


# In[148]:


from sklearn.metrics import mean_squared_log_error, mean_absolute_error

print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred))
print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred))


# In[149]:


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[153]:


#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()


# In[154]:


#Fit the model
model.fit(X_train, y_train)


# In[157]:


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[159]:


# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()


# In[150]:


#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[151]:


#Fit
GBR.fit(X_train, y_train)


# In[152]:


print("Accuracy --> ", GBR.score(X_test, y_test)*100)


# In[158]:


# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()


# In[ ]:




