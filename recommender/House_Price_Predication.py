# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

# Load dataset
housing = fetch_california_housing()

# Basic checks
print(type(housing))
print(housing)
print(housing.DESCR)
print(housing.feature_names)

# Convert to DataFrame
data = pd.DataFrame(housing.data, columns=housing.feature_names)
print(type(data))
print(data.head())

# Add target column
data['Price'] = housing.target

# Data info & checks
print(data.info())
print(data.isnull().sum())
print(data.describe())

# # Correlation matrix
# corr = data.corr()
# print(corr)
#
# # Heatmap for correlation (recommended)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()
#
# # Pairplot (can be heavy, so sample if system is slow)
# sns.pairplot(data.sample(500))   # use full data if system is strong
# plt.show()
#
# fig,ax=plt.subplots(figsize=(15,15))
# sns.boxplot(data=data,ax=ax)
# plt.show()

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_train_norm=scalar.fit_transform(x_train)

# fig,ax=plt.subplots(figsize=(15,15))
# sns.boxplot(data=x_train_norm,ax=ax)
# plt.show()

x_test_norm=scalar.transform(x_test)
# fig,ax=plt.subplots(figsize=(15,15))
# sns.boxplot(data=x_test_norm,ax=ax)
# plt.show()
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x_train_norm,y_train)
print(LR.coef_)
print(LR.intercept_)

regression_pred=LR.predict(x_test_norm)
residual=y_test-regression_pred
print(residual)
# sns.displot(residual,kind='kde')
# plt.show()

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_absolute_error(y_test,regression_pred))
print(mean_squared_error(y_test,regression_pred))
print(r2_score(y_test,regression_pred))