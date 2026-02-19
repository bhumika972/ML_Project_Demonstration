import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import DistanceMetric
import seaborn as sns
data=pd.read_csv(r"C:\Users\S.A COMPUTER\ML_Project_Demonstration\recommender\Regularization\Housing_Dataset.csv")

print(data.head())
print(data.columns)
print(data.shape)
print(data.isnull().sum())

usefull=['Suburb', 'Rooms', 'Type', 'Price',
         'Method', 'SellerG','Car','Bedroom2','Bathroom',
       'Landsize','Distance',
         'BuildingArea','CouncilArea',
         'Regionname', 'Propertycount']
newData=data[usefull]
print(newData.shape)
print(newData.isnull().sum())

cols=['Car','Bathroom','Bedroom2','Propertycount','Distance']
newData[cols]=newData[cols].fillna(0)


sns.displot(newData.BuildingArea)
plt.show()
newData['BuildingArea']=newData['BuildingArea'].replace(np.nan,newData['BuildingArea'].median())
newData['Landsize']=newData['Landsize'].replace(np.nan,newData['Landsize'].median())
print(newData.isnull().sum())
newData.dropna(inplace=True)
print(newData.isna().sum())

newData=pd.get_dummies(newData,drop_first=True)
print(newData.shape)

x=newData.drop(columns='Price')
y=newData['Price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

from sklearn.linear_model import LinearRegression


model=LinearRegression()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))
print(model.score(x_train,y_train))

from sklearn.linear_model import Lasso
lasso_model=Lasso(alpha=50,max_iter=100,tol=0.1)
lasso_model.fit(x_train,y_train)
print(lasso_model.score(x_test,y_test))
print(lasso_model.score(x_train,y_train))