# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:21:27 2020

@author: Ketan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

sal=pd.read_csv("C:\\Users\\Ketan\\Desktop\\Data Science\\simple linear regression\\Salary_Data.csv")
sal.info()
sal.describe()
sal.shape

# Measures of Central Tendency
np.mean(sal.YearsExperience) # 5.313
np.mean(sal.Salary) # 76003
np.median(sal.Salary) # 65237
np.median(sal.YearsExperience) # 4.7

# Measures of Dispersion
np.std(sal)
np.var(sal.YearsExperience) ##  7.78
np.var(sal.Salary) ##  726499261.73

# skewnesss
skew(sal)
# kurtosis
kurtosis(sal)

x=np.array(sal.Salary)
y=np.array(sal.YearsExperience)

# Normal Q-Q plot
plt.plot(sal);plt.legend(['Years_of_Experience','Salary_hike']); plt.show()

st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)

#Normal Probability Distribution

x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='blue');plt.xlim(np.min(x),np.max(x));plt.xlabel('Years_of_Experience');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color = 'orange');plt.xlim(np.min(y),np.max(y)) ;plt.xlabel('Salary_hike');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(sal.YearsExperience)
plt.hist(sal.Salary)

# Boxplot
plt.boxplot(sal["YearsExperience"])
plt.boxplot(sal["Salary"])
#help(sns.boxplot)
# another way ::::>>>
sns.boxplot(sal,color='coral',orient='v')
sns.boxplot(sal['YearsExperience'],orient='v',color='red') # orient = 'v' -> Vertival
sns.boxplot(sal['Salary'],orient='v',color='yellow')  # orient = 'h' ->horizontal

 # scatter plot
plt.scatter(x,y,label='Scatter_plot',color='r',s=40);plt.xlabel('Years_of_Experience');plt.ylabel('Saalry_hike');plt.title('Scatter Plot ');
 
np.corrcoef(x,y) ## array
sal.corr()
sns.heatmap(sal.corr())

sns.pairplot(sal)
sns.countplot(x)
sns.countplot(y)


# Simple Linear Regression Model
model = smf.ols('Salary~YearsExperience',data=sal).fit()
model.summary()  # R_sqr =0.957 
model.params

pred = model.predict(sal)
error = sal.Salary-pred
sum(error) ## 0

# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_of_Experiance');plt.title('Scatter Plot')
np.corrcoef(x,y) 
np.sqrt(np.mean(error**2))  # RMSE = 5592,044

#Simple Linear Regression Model, Apply Log transformation to X- variable
model1 = smf.ols('Salary~np.log(YearsExperience)',data=sal).fit()
model1.summary() # R_sqr = 0.854 
model1.params

pred1 = model1.predict(sal)
error1 = sal.Salary-pred1
sum(error1) ## 0

# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred1,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_of_Experiance');plt.title('Scatter Plot')
#help(plt.plot)
np.corrcoef(np.log(x),y) 
# RMSE
np.sqrt(np.mean(error1**2))  # RMSE = 10302.8937

# Simple Linear Regression Model2 , Apply Log transformation on 'Y'

model2 = smf.ols('np.log(Salary)~YearsExperience',data=sal).fit()
model2.summary() # R_sqr = 0.932 
model2.params

pred2= model2.predict(sal)
error2 = sal.Salary-np.exp(pred2)
sum(error2) ## 3309.35

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_of_Experiance');plt.title('Scatter Plot')
np.corrcoef(x,np.log(y)) 
# RMSE 
np.sqrt(np.mean(error2**2)) #RMSE = 7213.235


#### Residuals Vs Fitted values
plt.scatter(pred2,model2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# checking normal distribution for residual
plt.hist(model2.resid_pearson)
