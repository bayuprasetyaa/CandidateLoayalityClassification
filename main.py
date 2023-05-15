import os
import streamlit as st
import pandas as pd
import pickle


# Model and System
MAIN_PATH  = os.path.abspath(os.getcwd())
PATH_MODEL = os.path.join(MAIN_PATH, "model", "lgbm-1-classification.pkl")
print(MAIN_PATH)
lgbm = pickle.load(open(PATH_MODEL, 'rb'))
feature = pd.DataFrame({
    'city':[],
    'city_development_index':[],
    'relevent_experience':[],
    'enrolled_university':[],
    'education_level':[],
    'major_discipline':[],
    'experience':[],
    'company_size':[],
    'company_type':[],
    'last_new_job':[],
    'training_hours':[],
})


# Title
st.set_page_config(page_title="Loyal Candidate Prediction", initial_sidebar_state="collapsed")
st.header('Loyal Candidate Prediction')
st.write("""
         The Loyal Candidate Prediction App is a web-based application that helps employers predict the loyalty of their job candidates. 
         The app uses machine learning algorithms to analyze a candidate's professional history 
         and other relevant data points to determine their likelihood of staying with the company for an extended period.
         """)

# Name and image
name = st.text_input("Candidate Name")
img = st.file_uploader("Upload photo",type=['jpg', 'png', 'jpeg'])

# House feature
col1, col2 = st.columns(2)

with col1:
    st.subheader('Personal Information')
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        city = st.number_input(label="City Index", min_value=1, step=1)

        
    with subcol2:
        cdi = st.number_input(label="City Development Index", min_value=0.0, max_value=1.0, step=0.1)
    
    rev_exp = st.selectbox("Relevant Experience", ('Has relevent experience', 'No relevent experience'))
    enr_univ = st.selectbox("Enrolment Univercity", ('no_enrollment', 'Part time course', 'Full time course'))
    edu = st.selectbox("Education", ('Masters', 'Graduate', 'Phd'))
    major_dicipline = st.selectbox("Major Dicipline", ('STEM', 'Business Degree', 'Arts', 'No Major', 'Humanities','Other'))

with col2:
    st.subheader('Working History')
    exp = st.number_input(label="Experience (year)", min_value=0,step=1)
    cmp_size = st.selectbox("Company Size", ('10/49', '10000+', '100-500', '50-99', '1000-4999', '<10', '500-999', '5000-9999'))
    cmp_type = st.selectbox("Company Type", ('Pvt Ltd', 'Other', 'Early Stage Startup', 'NGO', 'Funded Startup', 'Public Sector'))
    last_job = st.selectbox("Last New Job", ('1', '>4', '2', 'never', '3', '4'))
    training_hour = st.number_input(label="Training Hours", min_value=0, step=1)

pred_process = st.button("Predict",use_container_width=True)

if pred_process:
    feature.loc[0, 'city'] = f"city_{city}"
    feature.loc[0,'city_development_index'] = cdi
    feature.loc[0,'relevent_experience'] = rev_exp
    feature.loc[0,'enrolled_university'] = enr_univ
    feature.loc[0,'education_level'] = edu
    feature.loc[0,'major_discipline'] = major_dicipline
    feature.loc[0,'experience'] = '>20' if exp > 20 else '<1' if exp < 1 else str(exp)
    feature.loc[0,'company_size'] = cmp_size
    feature.loc[0,'company_type'] = cmp_type
    feature.loc[0,'last_new_job'] = last_job
    feature.loc[0,'training_hours'] = training_hour
    
    print(feature)
    prob = lgbm.predict_proba(feature)

    photo, information = st.columns(2)
    
    with photo:
        st.image(img)
    
    with information:
        st.write("Name: ", name)
        st.write("Relevant Experience: ", rev_exp)
        st.write("Education Level: ", edu)
        st.write("Experience: ", str(exp), " year")
        st.write("Loyality Pred. : ", str(round(prob[:,0][0], 2)))