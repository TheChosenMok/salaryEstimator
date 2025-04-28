import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained regression model
with open("salary2025_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💼 Salary Predictor")
st.subheader("📈 Predict your salary based on skills, experience, education, and other features")

education_map = {'No formal education past high school': 1,
    'Some college/university study without earning a bachelor’s degree': 2,
    'Bachelor’s degree': 3,
    'Master’s degree': 4, 
    'Doctoral degree': 5, 
    'Professional doctorate': 6, 
    'I prefer not to answer': np.nan}

age_map = {
    '18-21': 1,
    '22-24': 2,
    '25-29': 3,
    '30-34': 4,
    '35-39': 5,
    '40-44': 6,
    '45-49': 7,
    '50-54': 8,
    '55-59': 9,
    '60-69': 10,
    '70+': 11, 
}

coding_experience_map = {
    'I have never written code': 1,
    '< 1 years': 2,
    '1-3 years': 3,
    '3-5 years': 4,
    '5-10 years': 5,
    '10-20 years': 6,
    '20+ years': 7,
}

company_size_map = {
    '0-49 employees': 1,
    '50-249 employees': 2,
    '250-999 employees': 3,
    '1000-9,999 employees': 4,
    '10,000 or more employees': 5
}

dataScience_size_map = {
    '0': 1,
    '1-2': 2,
    '3-4': 3,
    '5-9': 4,
    '10-14': 5,
    '15-19': 6,
    '20+': 7
}

# User input widgets
age = st.selectbox("What is your age group", list(age_map.keys()))
coding_experience = st.selectbox("For how many years have you been coding/programming", list(coding_experience_map.keys()))
company_size = st.selectbox("What is the size of your company where you are currently employed", list(company_size_map.keys()))
education = st.selectbox("What is the highest level of formal education you have attained or plan to attain within the next 2 years", list(education_map.keys()))
cuntry = st.selectbox("Which Country do you currently reside in", ["United States of America", "India", "Japan", "Brazil", "Nigeria",
    "Spain", "United Kingdom of Great Britain and Northern Ireland", "Mexico",
    "France", "Pakistan", "South Korea", "Canada", "Turkey", "Taiwan",
    "China", "Indonesia", "Russia", "Other"])
gender = st.selectbox("What is your gender?", ["Man", "Woman", "Other"])
number_of_data_scientists = st.selectbox("Approximately how many individuals are responsible for data science workloads at your place of business?", list(dataScience_size_map.keys()))

student_status = st.selectbox("Are you currently a student? (high school, university, or graduate)", ["Yes", "No"])

ml_incorporation = st.selectbox("Does your current employer incorporate machine learning methods into their business?", [
    "We are exploring ML methods (and may one day put a model into production)",
    "We use ML methods for generating insights (but do not put ML models into production)",
    "We have well established ML methods (i.e., models in production for more than 2 years)",
    "We recently started using ML methods (i.e., models in production for less than 2 years)",
    "No (we do not use ML methods)",
    "I do not know"
])

current_role = st.selectbox("Select the title most similar to your current role (or most recent title if retired):", [
    "Machine Learning/ MLops Engineer",
    "Data Architect",
    "Data Engineer",
    "Data Scientist",
    "Data Administrator",
    "Developer Advocate",
    "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
    "Manager (Program, Project, Operations, Executive-level, etc)",
    "Research Scientist",
    "Software Engineer",
    "Engineer (non-software)",
    "Statistician",
    "Teacher / professor",
    "Currently not employed",
    "Other"
])

industry = st.selectbox("In what industry is your current employer/contract (or your most recent employer if retired)?", [
    "Computers/Technology",
    "Accounting/Finance",
    "Broadcasting/Communications",
    "Academics/Education",
    "Energy/Mining",
    "Government/Public Service",
    "Insurance/Risk Assessment",
    "Online Service/Internet-based Services",
    "Marketing/CRM",
    "Manufacturing/Fabrication",
    "Medical/Pharmaceutical",
    "Non-profit/Service",
    "Retail/Sales",
    "Shipping/Transportation",
    "Other"
])

base_cols = [
    # numeric / ordinal
    'What is your age (# years)?',
    'Are you currently a student? (high school, university, or graduate)',
    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
    'For how many years have you been writing code and/or programming?',
    'What is the size of the company where you are employed?',
    'Approximately how many individuals are responsible for data science workloads at your place of business?',
    # gender dummies
    'What is your gender? - Selected Choice_Man',
    'What is your gender? - Selected Choice_Woman',
    'What is your gender? - Selected Choice_Other',
    # country dummies  (17)
    'In which country do you currently reside?_Brazil',
    'In which country do you currently reside?_Canada',
    'In which country do you currently reside?_China',
    'In which country do you currently reside?_France',
    'In which country do you currently reside?_India',
    'In which country do you currently reside?_Indonesia',
    'In which country do you currently reside?_Japan',
    'In which country do you currently reside?_Mexico',
    'In which country do you currently reside?_Nigeria',
    'In which country do you currently reside?_Other',
    'In which country do you currently reside?_Pakistan',
    'In which country do you currently reside?_Russia',
    'In which country do you currently reside?_South Korea',
    'In which country do you currently reside?_Spain',
    'In which country do you currently reside?_Taiwan',
    'In which country do you currently reside?_Turkey',
    'In which country do you currently reside?_United Kingdom of Great Britain and Northern Ireland',
    'In which country do you currently reside?_United States of America',
    # employer-ML dummies (6)
    'Does your current employer incorporate machine learning methods into their business?_We are exploring ML methods (and may one day put a model into production)',
    'Does your current employer incorporate machine learning methods into their business?_We use ML methods for generating insights (but do not put working models into production)',
    'Does your current employer incorporate machine learning methods into their business?_We have well established ML methods (i.e., models in production for more than 2 years)',
    'Does your current employer incorporate machine learning methods into their business?_We recently started using ML methods (i.e., models in production for less than 2 years)',
    'Does your current employer incorporate machine learning methods into their business?_No (we do not use ML methods)',
    'Does your current employer incorporate machine learning methods into their business?_I do not know',
    # role dummies (13)
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Data Administrator',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Data Analyst (Business, Marketing, Financial, Quantitative, etc)',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Data Architect',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Data Engineer',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Data Scientist',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Developer Advocate',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Engineer (non-software)',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Machine Learning/ MLops Engineer',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Manager (Program, Project, Operations, Executive-level, etc)',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Research Scientist',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Software Engineer',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Statistician',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Teacher / professor',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_Other',
    # industry dummies (15)
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Academics/Education',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Accounting/Finance',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Broadcasting/Communications',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Computers/Technology',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Energy/Mining',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Government/Public Service',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Insurance/Risk Assessment',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Manufacturing/Fabrication',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Marketing/CRM',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Medical/Pharmaceutical',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Non-profit/Service',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Online Service/Internet-based Services',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Retail/Sales',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Shipping/Transportation',
    'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_Other'
]

row = {c: 0 for c in base_cols}

row['What is your age (# years)?'] = age_map[age]
row['Are you currently a student? (high school, university, or graduate)'] = 1 if student_status == 'Yes' else 0
row['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'] = education_map[education]
row['For how many years have you been writing code and/or programming?'] = coding_experience_map[coding_experience]
row['What is the size of the company where you are employed?'] = company_size_map[company_size]
row['Approximately how many individuals are responsible for data science workloads at your place of business?'] = dataScience_size_map[number_of_data_scientists]
#flip it
row[f'What is your gender? - Selected Choice_{gender}'] = 1

row[f'In which country do you currently reside?_{cuntry}'] = 1

row[f'Does your current employer incorporate machine learning methods into their business?_{ml_incorporation}'] = 1

row[f'Select the title most similar to your current role (or most recent title if retired): - Selected Choice_{current_role}'] = 1

row[f'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice_{industry}'] = 1

X = pd.DataFrame([row])[model.feature_names_in_]   # ensures exact order

#6. Predict 
if st.button("Predict Salary"):
    pred = model.predict(X)[0]
    st.success(f"Estimated yearly compensation: **${pred:,.0f}**")
    
