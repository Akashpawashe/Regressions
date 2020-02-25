
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

calory= pd.read_csv("D:\\excelR\\Data science notes\\simple linear regression\\asgmnt\\calories_consumed.csv")
calory.info()
calory.describe()
calory.shape
calory.columns
calory.corr()

# Changing the Column names
calory.rename(columns={'WeightGained':'wt'},inplace =True)
calory.rename(columns={'CaloriesConsumed':'cal'},inplace=True)

# Measures of Central Tendency
np.mean(calory.wt) # 357.71
np.mean(calory.cal) # 2340.71
np.median(calory.wt) # 200
np.median(calory.cal) # 2250

# Measures of Dispersion
np.std(calory)
np.var(calory)

# skewnesss
skew(calory)
# kurtosis
kurtosis(calory)

x=np.array(calory.cal)
y=np.array(calory.wt)

# Normal Q-Q plot
plt.plot(calory);plt.legend(['Calories Consumed','Weight gained (grams)'])
plt.show()

st.probplot(x,dist='norm',plot=pylab)
st.probplot(y,dist='norm',plot=pylab)

#Normal Probability Distribution

x1 = np.linspace(np.min(x),np.max(x))
y1 = st.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='blue');plt.xlim(np.min(x),np.max(x));plt.xlabel('Calories_Consumed');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = st.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color = 'green');plt.xlim(np.min(y),np.max(y)) ;plt.xlabel('Weight_gained(grams)');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(calory.cal)
plt.hist(calory.wt)

# Boxplot
plt.boxplot(calory["wt"])
plt.boxplot(calory["cal"])
help(sns.boxplot)
# another way ::::>>>
sns.boxplot(calory,color='coral',orient='v')
sns.boxplot(calory['cal'],orient='v',color='red') # orient = 'v' -> Vertival
sns.boxplot(calory['wt'],orient='v',color='orange')  # orient = 'h' ->horizontal

 # scatter plot
plt.scatter(x,y,label='Scatter_plot',color='r',s=40);plt.xlabel('Calories_Consumed');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot ');
 
np.corrcoef(x,y) ## array
calory.corr()
sns.heatmap(calory.corr())

sns.pairplot(calory)
sns.countplot(x)
sns.countplot(y)
###  simple Regression model
model=smf.ols("wt ~ cal",data=calory).fit()
model.params
model.summary() ## R_sqr= 0.897
pred= model.predict(calory)
error= calory.wt- pred
sum(error) # 0
#np.mean(error) # 0

# Scatter plot between 'x' and 'y'
plt.scatter(x,y,color='red',s=40);plt.plot(x,pred,color='black');plt.xlabel('Calories_Consumed');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot')
# Correlation Coefficients
np.corrcoef(x,y) # r = 0.94699
np.sqrt(np.mean(error**2)) #RMSE value = 103.30

## simple Regression model1 , Apply log transformation on x-variables
model1 = smf.ols('wt~np.log(cal)',data=calory).fit()
model1.summary()  # R-sqr = 0.808
model1.params
pred1 = model1.predict(calory)
error1 =calory.wt-pred1
sum(error1) # 0

# Scatter Plot between log(x) and y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred1,color='black');plt.xlabel('log(Calories_Consumed)');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot')
# Correlation coefficient (r)
np.corrcoef(np.log(x),y)  # r = 0.8987 
np.sqrt(np.mean(error1**2)) #RMSE value = 141.0054

## simple Regression Model2 , Apply log transformation to Y-variable
model2 = smf.ols('np.log(wt)~cal',data=calory).fit()
model2.summary()  # R_sqr =0.878 
model2.params
pred2 = model2.predict(calory)
error2 = calory.wt-np.exp(pred2)
# Sum of Errors should be Zero
sum(error2)  # 73.78

# Scatter Plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred2,color='black');plt.xlabel('Calories_Consumed');plt.ylabel('log(Weight_gained(grams))');plt.title('Scatter Plot')
# Correlation Coefficient
np.corrcoef(x,np.log(y)) # 0.9368 
np.sqrt(np.mean(error2**2)) #RMSE value = 118.045



#### Residuals Vs Fitted values
plt.scatter(pred2,model2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# checking normal distribution for residual
plt.hist(model2.resid_pearson)

