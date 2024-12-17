import joblib
import streamlit as st
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie
from sklearn.preprocessing import StandardScaler

sc = joblib.load(open('scaler.pkl', 'rb'))





st.set_page_config(page_title='Water Quality Prediction', page_icon='Water Potability Logo.png', initial_sidebar_state='collapsed')

choose_model=st.sidebar.radio("Select The Prediction Model :",['Logistic Regression','SVM','Decision Tree','Random Forest','XG boost'])

if choose_model=='Logistic Regression':
     model_deploy = joblib.load(open('lr_model', 'rb'))
elif choose_model=='SVM':
     model_deploy = joblib.load(open('svm_model', 'rb'))
elif choose_model=='Decision Tree':
     model_deploy = joblib.load(open('dt_model', 'rb'))
elif choose_model=='Random Forest':
     model_deploy = joblib.load(open('rf_model', 'rb'))
elif choose_model=='XG boost':
     model_deploy = joblib.load(open('xg_model', 'rb'))



def predict(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes):
    features = np.array([float(ph), float(Hardness), float(Solids), float(Chloramines), float(Sulfate), float(Conductivity), float(Organic_carbon), float(Trihalomethanes)]).reshape(1, -1)
    scaled_features = sc.transform(features)
    prediction = model_deploy.predict(scaled_features)
    return prediction

choose = st.sidebar.selectbox("Menu",["Home", "Models"])





if choose == "Home":
    st.markdown("<h1 style='text-align: center;'>Water Quality Prediction</h1>", unsafe_allow_html=True)
    st.info('An application to predict if water is potable or not')
    st.write('---')
    
    st.subheader('Please Enter Your Data: ')

   
    ph=st.number_input('Ph')
    Hardness=st.number_input('Hardness')
    Solids=st.number_input('Solids')
    Chloramines=st.number_input('Chloramines')
    Sulfate=st.number_input('Sulfate')
    Conductivity=st.number_input('Conductivity')
    Organic_carbon=st.number_input('Organic Carbon')
    Trihalomethanes=st.number_input('Trihalomethanes')

    # Prediction
    result=predict(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes)
    if st.button('Predict'):
        if result==0:
            st.error('The water is not potable')
        else:
            st.success('The water is potable')
            st.balloons()
    
elif choose == "Models":
    st.markdown("<h1 style='text-align: center;'>The Used Models</h1>", unsafe_allow_html=True)
    st.write('---')
    st.subheader('Please Choose The Model: ')
    models = st.radio("The Models", ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'XG boost'])
    if models == 'Logistic Regression':
        st.write('### The accuracy is 50.375%')
        st.write('---')
        st.write('### Confusion Matrix :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cm_lr.png')
        st.write('---')
        st.write('### Classification Report :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cr_lr.png')
    elif models == 'SVM':
        st.write('### The accuracy is 52.375%')
        st.write('---')
        st.write('### Confusion Matrix :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cm_svm.png')
        st.write('---')
        st.write('### Classification Report :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cr_svm.png')  
    elif models == 'Decision Tree':
        st.write('### The accuracy is 71.125%')
        st.write('---')
        st.write('### Confusion Matrix :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cm_dt.png')
        st.write('---')
        st.write('### Classification Report :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cr_dt.png') 
    elif models == 'Random Forest':
        st.write('### The accuracy is 80.375%')
        st.write('---')
        st.write('### Confusion Matrix :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cm_rf.png')
        st.write('---')
        st.write('### Classification Report :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cr_rf.png') 
    elif models == 'XG boost':
        st.write('### The accuracy is 75.125%')
        st.write('---')
        st.write('### Confusion Matrix :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cm_xg.png')
        st.write('---')
        st.write('### Classification Report :')
        st.markdown("<br>", unsafe_allow_html=True)
        st.image('cr_xg.png') 


        


        
     



