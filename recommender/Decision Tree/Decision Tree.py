from random import sample

import pandas as pd
data=pd.read_csv("winequality-red.csv")
print(data.head())
print(data.columns)
print(set(data['quality']))
print(data.shape)          # Rows & columns
print(data.info())         # Data types & null values
print(data.describe())     # Statistical summary
print(data.isnull().sum())
import matplotlib.pyplot as plt

# data['quality'].value_counts().sort_index().plot(kind='bar')
# plt.xlabel("Quality Score")
# plt.ylabel("Count")
# plt.title("Wine Quality Distribution")
# plt.show()
#
# import seaborn as sns
# plt.figure(figsize=(12,8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix")
# plt.show()
# data.hist(figsize=(15,10), bins=20)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(12,6))
# sns.boxplot(data=data)
# plt.xticks(rotation=90)
# plt.show()
# Calculate IQR for chlorides
Q1 = data['chlorides'].quantile(0.25)
Q3 = data['chlorides'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove outliers
data = data[(data['chlorides'] >= lower_limit) &
            (data['chlorides'] <= upper_limit)]

print("Shape after removing outliers:", data.shape)
X = data.drop('quality', axis=1)
y = data['quality']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame (optional but better for understanding)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model._fit(X_train,y_train)
y_pred=model.predict(X_test)

# from sklearn import  tree
# plt.figure(figsize=(15,15))
# tree.plot_tree(model,filled=True)
# plt.show()
sampleData=data.head(20)
x_sample=sampleData.drop('quality', axis=1)
y = sampleData['quality']
sampleModel=DecisionTreeClassifier()
sampleModel.fit(x_sample,y)
print(set(sampleData['quality']))
from sklearn import  tree
plt.figure(figsize=(15,15))

tree.plot_tree(sampleModel,filled=True)
plt.show()

