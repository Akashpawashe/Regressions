

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import spicy.stats as st
import pylab

emp= pd.read_csv("D:\\excelR\\Data science notes\\simple linear regression\\asgmnt\\emp_data.csv")
emp.info()
emp.describe()
emp.shape
emp.rename(columns={'Salary_hike':'salary'},inplace=True)
emp.rename(columns={'Churn_out_rate':'ch_rate'},inplace=True)
emp.describe()

# Measures of Central Tendency
np.mean(emp)
np.median(emp.salary) # 1675
np.median(emp.ch_rate) # 71

# Measures of Dispersion
np.var(emp)
np.std(emp)

# Skewness and Kurtosis
skew(emp.salary) # 0.7238
skew(emp.Ch_rate) # 0.5457

kurtosis(emp.salary) # -0.4516
kurtosis(emp.ch_rate)# -0.7311


x = np.array(emp.salary)
y = np.array(emp.ch_rate)

# Normal Q-Q plot
plt.plot(emp.salary)
plt.plot(emp.ch_rate)

plt.plot(emp);plt.legend(['salary','ch_rate']);

st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)

# Normal Probability Distribution plot 

x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Salary_Hike');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color='blue');plt.xlim(np.min(y),np.max(y));plt.xlabel('Churn_out_rate');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(emp['salary'],color='coral')

plt.hist(emp['Ch_rate'],color='skyblue')

# Boxplot 
sns.boxplot(emp,orient='v')
sns.boxplot(emp['salary'],orient = 'v',color='coral')
sns.boxplot(emp['Ch_rate'],orient = 'v',color='skyblue')

sns.pairplot(emp)
sns.countplot(emp['salary'])
sns.countplot(emp['Ch_rate'])

# Scatter plot
plt.scatter(x,y,label='Scatter plot',color='coral',s=20);plt.xlabel('Salary_Hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot');
np.corrcoef(emp['salary'],emp['Ch_rate']) 
emp.corr()
sns.heatmap(emp.corr())

#Simple Linear Regression Model
model = smf.ols('Ch_rate~salary',data=emp).fit()
model.summary()  # R_sqr =0.831
model.params

pred = model.predict(emp)
error = emp.Ch_rate-pred
sum(error) # 0

# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
np.corrcoef(x,y) 
np.sqrt(np.mean(error**2))  # RMSE = 3.9975

# Simple Linear Regression Model2, Apply Log transformation to X- variable
model = smf.ols('Ch_rate~np.log(salary)',data=emp).fit()
model.summary() # R-sqr = 0.849 
model.params

pred1 = model1.predict(emp)
error1 = emp.Ch_rate-pred1
sum(error1)

# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred1,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
help(plt.plot)
np.corrcoef(np.log(x),y) 

# RMSE
np.sqrt(np.mean(error1**2))  # RMSE = 50250.40
# Simple Linear Regression Model3 , Apply Log transformation on 'Y'

model2 = smf.ols('np.log(Ch_rate)~salary',data=emp).fit()
model2.summary() # R_sqr = 0.874 
model2.params

pred2 = model2.predict(emp)
error2 = emp.Ch_rate-np.exp(pred2)
sum(error2)

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
np.corrcoef(x,np.log(y))

# RMSE 
np.sqrt(np.mean(error2**2)) #RMSE = 50006.50


#### Residuals Vs Fitted values
plt.scatter(pred2,model2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# checking normal distribution for residual
plt.hist(model2.resid_pearson)
