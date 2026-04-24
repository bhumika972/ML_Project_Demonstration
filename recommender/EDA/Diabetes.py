import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
data = pd.read_csv(r"C:\Users\S.A COMPUTER\ML_Project_Demonstration\recommender\EDA\diabetes.csv")
print(data.head())
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.describe())
print(data.isnull().sum())

# -------------------------------
# 2️⃣ Outlier Removal
# -------------------------------
q = data['BloodPressure'].quantile(0.99)
data = data[data['BloodPressure'] < q]

q = data['Insulin'].quantile(0.95)
data = data[data['Insulin'] < q]

print("Quantile threshold for Insulin:", q)

# -------------------------------
# 3️⃣ Replace zeros with median
# -------------------------------
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    print(col, (data[col] == 0).sum())
    data[col] = data[col].replace(0, data[col].median())

# -------------------------------
# 4️⃣ Visualization
# -------------------------------
fig, ax = plt.subplots(figsize=(15,15))
sns.boxplot(data=data, ax=ax)
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot=True)
plt.show()

print("Target distribution:\n", data.Outcome.value_counts())

# -------------------------------
# 5️⃣ Scaling
# -------------------------------
scaler = StandardScaler()
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 6️⃣ Train-Test Split
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# 7️⃣ Logistic Regression Model
# -------------------------------
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_probability = model.predict_proba(x_test)

# -------------------------------
# 8️⃣ Prediction Comparison
# -------------------------------
predication = pd.DataFrame({"Actual Data": y_test.values, "Predict Data": y_pred})
print(predication)

# -------------------------------
# 9️⃣ Model Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
