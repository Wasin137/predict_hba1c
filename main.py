import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle



def pre_processing(sbp, dbp, weight, height, age, sex, edu):
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)

    # Create dictionary of input data
    data_input = {'mean_sbp': sbp, 'mean_dbp': dbp, 'bmi': bmi, 'age': age, 'SEX': sex, 'EDUCATION': edu}

    # Convert dictionary to DataFrame
    pre_data = pd.DataFrame.from_dict(data_input, orient='index', columns=['Data'])
    pre_data = pre_data.T

    # Return DataFrame
    return pre_data

def std_encoded(pre_data):

    with open('model_rf.pkl', 'rb') as f:
        model_rf = pickle.load(f)

    with open('scaler_rf.pkl', 'rb') as f:
        scaler = pickle.load(f)
    features_to_scale = ['mean_sbp', 'mean_dbp', 'bmi', 'age']
    scaled_data = scaler.transform(pre_data[['mean_sbp', 'mean_dbp', 'bmi', 'age']])
    pre_data[features_to_scale] = scaled_data
    coded_dict = {'SEX_Female':0, 'SEX_Male':0,
       "EDUCATION_Associate's Degree":0, "EDUCATION_Bachelor's Degree":0,
       'EDUCATION_High School':0, 'EDUCATION_Master/Doctorate Degree':0,
       'EDUCATION_No':0, 'EDUCATION_No Data':0, 'EDUCATION_Pre Kindergarten':0,
       'EDUCATION_Primary School':0}
    if pre_data['SEX'][0] == 'Male':
        coded_dict['SEX_Male'] = 1
    elif pre_data['SEX'][0] == 'Female':
        coded_dict['SEX_Female'] = 1
    if pre_data['EDUCATION'][0] == 'No':
        coded_dict['EDUCATION_No'] = 1
    elif pre_data['EDUCATION'][0] == 'Pre Kindergarten':
        coded_dict['EDUCATION_Pre Kindergarten'] = 1
    elif pre_data['EDUCATION'][0] == 'Primary School':
        coded_dict['EDUCATION_Primary School'] = 1
    elif pre_data['EDUCATION'][0] == 'High School':
        coded_dict['EDUCATION_High School'] = 1
    elif pre_data['EDUCATION'][0] == "Associate's Degree":
        coded_dict["EDUCATION_Associate's Degree"] = 1
    elif pre_data['EDUCATION'][0] == "Bachelor's Degree":
        coded_dict["EDUCATION_Bachelor's Degree"] = 1
    elif pre_data['EDUCATION'][0] == 'Master/Doctorate Degree':
        coded_dict['EDUCATION_Master/Doctorate Degree'] = 1
    else:
        coded_dict['EDUCATION_No Data'] = 1
    
    coded_df = pd.DataFrame.from_dict(coded_dict, orient='index', columns=['Data'])
    coded_df = coded_df.T

    predicting_data = pd.concat([pre_data[features_to_scale], coded_df], axis=1)
    
    predicted = model_rf.predict(predicting_data)
    hba1c = predicted[0]
    return hba1c.round(2)


st.title('Prediciting :red[HbA1C] Level')
st.header('Testing Version (V 0.1.1)')
st.subheader('Last updated :blue[16 Mar 2023]')

with st.sidebar:
    st.write('RandomForest Regressor Model (Scikit-learn)')
    st.write('Parameters tuning')
    st.write('max_depth 5')
    st.write('min_samples_split 2')
    st.write('n_estimators=100')
    st.write('Hatyai Hospital & Hatyai HDC Database')
    st.write('11,758 Records')
    st.write('Contact: :blue[wasin.kamp@gmail.com]')

col1, col2 = st.columns(2)
with col1:
    sbp = st.number_input('Systolic blood pressure')
    dbp = st.number_input('Diastolic blood pressure')
    weight = st.number_input('Weight (kg.)')
    height = st.number_input('Height (cm.)')
    age = st.number_input('Age')
    

with col2:
    sex = st.radio('Sex', ('Male', 'Female'))
    edu = st.radio('Hightest Education', ('No' ,'Pre Kindergarten', 'Primary School', 'High School', "Associate's Degree", "Bachelor's Degree", 'Master/Doctorate Degree', 'No Data'))

if st.button('Predict!'):
    pre_data = pre_processing(sbp, dbp, weight, height, age, sex, edu)
    st.write(pre_data)
    predicit_data = std_encoded(pre_data)
    st.write(f'Your prediction for HbA1c is :red[**{predicit_data}**]')

