import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data=pd.read_csv(r"C:\Users\S.A COMPUTER\ML_Project_Demonstration\recommender\EDA\diabetes.csv")
print(data.head())
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.describe())
print(data.isnull().sum())

# fig,ax=plt.subplots(figsize=(15,15))
# sns.boxplot(data=data,ax=ax)
# plt.show()
q=data['BloodPressure'].quantile(0.99)
data=data[data['BloodPressure']<q]
q=data['Insulin'].quantile(0.95)
data=data[data['Insulin']<q]
# fig,ax=plt.subplots(figsize=(12,12))
# sns.boxplot(data=newdata,ax=ax)
# plt.show()
print(q)


cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for col in cols:
    print(col, (data[col] == 0).sum())

for col in cols:
    data[col] = data[col].replace(0, data[col].median())

fig,ax=plt.subplots(figsize=(15,15))
sns.boxplot(data=data,ax=ax)
plt.show()

print("Final shape:", data.shape)

plt.figure(figsize=(10,10))
ax=sns.heatmap(data.corr(),annot=True)
plt.show()
print(data.Outcome.value_counts())