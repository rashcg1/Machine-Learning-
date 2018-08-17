# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:11:08 2018

@author: Rashu
"""

import numpy as ny
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


#1. Loading the data and simple analysis of attributes.
#import the data
train=pd.read_csv(r"D:\HousePredictions\train.csv")
test=pd.read_csv(r"D:\HousePredictions\test.csv")

#Check first few rows of the data
train.head()
test.head()

#to know the number samples use shape attribute(shape is not a method and hence no ()) of pandas
train.shape
test.shape

#Summary of all the columns
train.describe()
test.describe()
#after checking headers of the dataset, only difference between training and test dataset is the sales price.


# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)



#2.Analysising test column(sales price)-Univariate analysis.
#plotting histogram and to know if histogram is normally distributed skewness and kurtosis measures are important
#Follow the "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm" to know more about skewness and kurtosis

#using seaborn sns to plot histogram

sns.distplot(train['SalePrice'],fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu={:.2f} and sigma={:.2f}\n'.format(mu,sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()

from scipy import stats

res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#to know the skewness and kurtosis
print('the skewness is ', train['SalePrice'].skew())
print('the kurtosis is ',train['SalePrice'].kurt())



#3). Multivariable analysis.


