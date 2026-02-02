import pandas as pd


# Loading Data: This operation reads data from files such as CSV, Excel or JSON into a DataFrame
df=pd.read_csv(r"C:\Users\S.A COMPUTER\ML_Project_Demonstration\data\student_study_habit.csv")
print(df.head())
# Viewing and Exploring Data: After loading data, it is important to understand its structure and content.
# This methods allow you to inspect rows, summary statistics and metadata.
print(df.info())
print(df.isnull().sum())

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

df = df.drop(columns=["timestamp"])
df['study_hours']=pd.to_numeric(df['study_hours'],errors="coerce")

categorical_cols = ['age_group', 'gender', 'education_level', 'study_time', 'fixed_schedule',
                    'study_method', 'take_notes', 'study_device', 'biggest_distraction',
                    'social_media_breaks', 'concentration_level', 'motivation',
                    'satisfied', 'study_challenge']

for col in categorical_cols:
    df[col] = df[col].astype('category')

print(df.info())
