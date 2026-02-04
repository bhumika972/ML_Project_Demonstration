import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Loading Data: This operation reads data from files such as CSV, Excel or JSON into a DataFrame
df=pd.read_csv(r"C:\Users\S.A COMPUTER\ML_Project_Demonstration\data\student_study_habit.csv")
print(df.head())
# Viewing and Exploring Data: After loading data, it is important to understand its structure and content.
# This methods allow you to inspect rows, summary statistics and metadata.
print(df.info())


df.columns = df.columns.str.strip()

df = df.rename(columns={
    "Timestamp": "timestamp",
    "What is your age?": "age_group",
    "What is your gender?": "gender",
    "What is your current level of education?": "education_level",
    "How many hours do you study daily (on average)?": "study_hours",
    "At what time do you usually study?": "study_time",
    "Do you follow a fixed study schedule?": "fixed_schedule",
    "Which study method do you use most often?": "study_method",
    "Do you take notes while studying?": "take_notes",
    "Which device do you mostly use for studying?": "study_device",
    "What is your biggest distraction during study time?": "biggest_distraction",
    "Do you use social media during study breaks?": "social_media_breaks",
    "How would you rate your concentration level while studying?": "concentration_level",
    "What motivates you to study?": "motivation",
    "Are you satisfied with your current study habits?": "satisfied",
    "What is the biggest challenge you face in studying?": "study_challenge"
})

print("Print all columns names\n",df.columns)
print(df.info())

study_hours_ordered = ["Less than 1 hour", "1–2 hours", "3–4 hours", "More than 4 hours"]
df["study_hours"] = pd.Categorical(df["study_hours"], categories=study_hours_ordered, ordered=True)

categorical_cols = ['age_group', 'gender', 'education_level', 'study_time', 'fixed_schedule',
                    'study_method', 'take_notes', 'study_device', 'biggest_distraction',
                    'social_media_breaks', 'concentration_level', 'motivation',
                    'satisfied', 'study_challenge']

for col in categorical_cols:
    df[col] = df[col].astype('category')

df.drop(columns=["timestamp"], inplace=True)

print(df.info())

print(df.isnull().sum())


categorical_cols = df.select_dtypes(include="category").columns


for col in categorical_cols:
    df[col]=df[col].fillna(df[col].mode()[0],inplace=True)

print(df.isnull().sum())
print(df.head(47))



en_data = df[["gender", "fixed_schedule", "satisfied"]]
print(en_data.dtypes)


ohe = OneHotEncoder()

array = ohe.fit_transform(en_data)

encoded_df = pd.DataFrame(
    array,
    columns=ohe.get_feature_names_out(en_data.columns)
)

print(encoded_df)
print(encoded_df.info())



#
# age_order = {
#     "Below 18": 0,
#     "19–22": 1,
#     "23–25": 2,
#     "Above 25": 3
# }
# df["age_group"] = df["age_group"].map(age_order)
#
# study_hours_order = {
#     "Less than 1 hour": 0,
#     "1–2 hours": 1,
#     "3–4 hours": 2,
#     "More than 4 hours": 3
# }
# df["study_hours"] = df["study_hours"].map(study_hours_order)
#
#
# concentration_order = {
#     "Very poor": 0,
#     "Poor": 1,
#     "Average": 2,
#     "Good": 3,
#     "Very good": 4
# }
# df["concentration_level"] = df["concentration_level"].map(concentration_order)
#
# nominal_cols = [
#     "gender", "study_method", "study_device",
#     "biggest_distraction", "motivation", "study_challenge", "study_time"
# ]
# df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
#
# df["fixed_schedule"] = df["fixed_schedule"].map({
#     "Yes": 1,
#     "No": 0,
#     "Sometimes": 0.5   # industry trick: partial behavior
# })
