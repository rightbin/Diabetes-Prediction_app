import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras 
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

from eda_app import run_eda_app
from ml_app import run_ml_app


def main():
    st.title('당뇨병 예측 서비스입니다.')
    st.write("왼쪽 사이드바를 이용하여 메뉴를 선택하세요.")

    # 사이드바 메뉴
    menu= ['Home','EDA', 'ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write('이 앱은 Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,bmi,Dpf,Age의 정보를 입력 받습니다.')
        st.write('이 앱은 Random Forest를 이용한 예측 모델입니다. 기초 자료로만 활용해주세요.')
        st.write('왼쪽의 사이드바에서 선택하세요.')
        st.image('data/nurse.jpg')

    elif choice == 'EDA':
        run_eda_app()

    elif choice =='ML':
        run_ml_app()



if __name__ == '__main__':
    main()