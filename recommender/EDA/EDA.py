import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"C:\Users\S.A COMPUTER\ML_Project_Demonstration\data\housing.csv")
print(data)
print(type(data))
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.describe())
data['mainroad']=data['mainroad'].astype('category')
data['guestroom']=data['guestroom'].astype('category')
data['basement']=data['basement'].astype('category')
data['airconditioning']=data['airconditioning'].astype('category')
data['prefarea']=data['prefarea'].astype('category')
data['furnishingstatus']=data['furnishingstatus'].astype('category')
data['hotwaterheating']=data['hotwaterheating'].astype('category')
#print(data.corr())

print(data.info())

dataframe=pd.get_dummies(data,drop_first=True)
print(dataframe.columns)
corr=dataframe.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
dataframe[['area','bedrooms','bathrooms','stories','parking']]=scalar.fit_transform(dataframe[['area','bedrooms','bathrooms','stories','parking']])
print(dataframe.head(4))
print(dataframe.describe())

from sklearn.model_selection import train_test_split
x=dataframe.drop('price',axis=1)
y=dataframe['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model=LinearRegression()

model.fit(x_train,y_train)
predict=model.predict(x_test)
print(predict)

mse=mean_squared_error(y_test,predict)
print(mse)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predict)
r2 = r2_score(y_test, predict)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Model Accuracy (%):", r2 * 100)











