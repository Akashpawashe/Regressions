# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:30:08 2020

@author: Ketan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

time=pd.read_csv("C:\\Users\\Ketan\\Desktop\\Data Science\\simple linear regression\\delivery_time.csv")
time.info()
time.rename(columns={'Delivery Time':'dt'},inplace=True)
time.rename(columns={'Sorting Time':'st'},inplace=True)
time.describe()
time.shape

# Measures of Central Tendency
np.mean(time)
np.median(time.dt) # 17.83
np.median(time.st) # 6

# Measures of Dispersion
np.var(time)
np.std(time)

# Skewness and Kurtosis
skew(time.st) # 0.0436
skew(time.dt) #  0.326

kurtosis(time.st) # -1.16
kurtosis(time.dt) # -0.02

x = np.array(time.st)
y = np.array(time.dt)

# Normal Q-Q plot
plt.plot(time);plt.legend(['Delivery_time','Sorting_time']);

st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)

# Normal Probability Distribution 
x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(y))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Sorting_Time');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color='blue');plt.xlim(np.min(y),np.max(y));plt.xlabel('Delivery_Time');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(time['st'],color='blue')
plt.hist(time['dt'],color = 'red')

# Boxplot 
sns.boxplot(time,orient='v')
sns.boxplot(time['st'],orient = 'v',color='coral')
sns.boxplot(time['dt'],orient = 'v',color='yellow')

sns.pairplot(time)
sns.countplot(time['st'])
sns.countplot(time['dt'])

# Scatter plot
plt.scatter(x,y,label='Scatter Plot',color='blue',s=20);plt.xlabel('Sorting_Time');plt.ylabel('Delivery_Time');plt.title('Scatter Plot')

np.corrcoef(time['st'],time['dt'])
time.corr()
sns.heatmap(time.corr())

#  Simple Linear Regression Model1
model = smf.ols('dt~st',data=time).fit()
model.summary()  # R_sqr =0.682 
model.params

pred = model.predict(time)
error = time.dt-pred
sum(error) # 0

# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
np.corrcoef(x,y)  
np.sqrt(np.mean(error**2))  # RMSE = 2.79165

# Simple Linear Regression Model2, Apply Log transformation to X- variable
model1 = smf.ols('dt~np.log(st)',data=time).fit()
model1.summary() # R-sqqr = 0.695 
model1.params

pred1 = model1.predict(time)
error1 = time.dt-pred1
sum(error1) # 0
# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred1,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
help(plt.plot)
np.corrcoef(np.log(x),y)
np.sqrt(np.mean(error1**2))  # RMSE = 2.733

# Simple Linear Regression Model3 , Apply Log transformation on 'Y'
model2 = smf.ols('np.log(time.dt)~st',data=time).fit()
model2.summary() # R_sqr = 0.711 
model2.params

pred2 = model2.predict(time)
error2= time.dt-np.exp(pred2)
sum(error2) # 4.160

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred2,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
np.corrcoef(x,np.log(y)) 
np.sqrt(np.mean(error2**2)) #RMSE = 2.940


#### Residuals Vs Fitted values
plt.scatter(pred2,model2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# checking normal distribution for residual
plt.hist(model2.resid_pearson)
