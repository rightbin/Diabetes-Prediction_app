import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from PIL import Image,ImageFilter,ImageEnhance
import h5py
import tensorflow.keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle 
import joblib
from sklearn.ensemble import RandomForestClassifier


def run_ml_app():
    st.subheader('당신의 수치를 입력해주세요')

    # Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
    # Pregnancies : Number of times pregnant 임신횟수

    # Glucose : 공복혈당 Plasma glucose concentration a 2 hours in an oral glucose tolerance test

    # BloodPressure : Diastolic blood pressure (mm Hg)

    # SkinThickness : Triceps skin fold thickness (mm)

    # Insulin : 2-Hour serum insulin (mu U/ml)

    # BMI : Body mass index (weight in kg/(height in m)^2)

    # Dpf Diabetes pedigree function

    # Age (years)

    # COutcome : class variable (0 or 1) 268 of 768 are 1, the others are 0

    st.write('Pregnancies는 임신횟수입니다.')
    Pregnancies = st.slider("Pregnancies",0,17)
    st.write("임신 횟수는 {} 번 입니다.".format(Pregnancies))

    
    st.write('Glucose는 공복혈당입니다.')
    Glucose = st.slider("Glucose",1,199)
    st.write("공복혈당는 {} 입니다.".format(Glucose))

    
    st.write('BloodPressure는 혈압입니다.')
    BloodPressure = st.slider("BloodPressure",1,122)
    st.write("혈압은 {} (mm Hg) 입니다.".format(BloodPressure))

    
    st.write('SkinThickness는 피부 두께입니다.')
    SkinThickness = st.slider("SkinThickness",1,99)
    st.write("피부 두께는 {} (mm) 입니다.".format(SkinThickness))

    
    st.write('Insulin는 인슐린입니다.')
    Insulin = st.slider("Insulin",1,846)
    st.write("인슐린은 {} 번 입니다.".format(Insulin))

    
    st.write('BMI는 BMI지수입니다.')
    bmi = st.slider("BMI",1,68)
    st.write("BMI지수는 {} (weight in kg/(height in m)^2) 입니다.".format(bmi))


    
    st.write('Dpf는 당뇨병 혈통 기능입니다.')
    Dpf = st.slider("Dpf",1.0,3.0)
    st.write(" 당뇨병 혈통 기능는 {} 입니다.".format(Dpf))

    
    st.write('Age는 나이입니다.')
    Age = st.slider("Age",1,100)
    st.write("나이는 {} 살 입니다.".format(Age))

    #예측한다.

    # model = pickle.load(open("data/pima.pickle.dat", "rb"))
    
    model = joblib.load('data/best_model.pkl')   
    new_data = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,bmi,Dpf,Age])

    #new_data = np.array([0, 0.36  , 0.875 , 0.09547739, 0.48979592])
    
    new_data = new_data.reshape(1,-1)
    
    # sc_X = joblib.load('data/sc_X.pkl')

    # new_data = sc_X.transform(new_data)

    y_pred = model.predict(new_data)

    #st.write(predicted_data[0][0])
    
    # sc_y = joblib.load('data/sc_y.pkl')

    # y_pred_original = sc_y.inverse_transform( y_pred)

    if st.button("예측 결과 확인하기"):
        
        if y_pred == [0]:
            st.write("예측결과는 당뇨일 가능성이 적습니다.")            
        else :
            st.write("예측결과는 당뇨일 가능성이 높습니다. 의사와 상담하세요.")
        
        # st.write(  "에측 결과입니다. {:,.1f} 달러의 차를 살 수 있습니다".format(y_pred_original[0][0])  )


