
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from google.colab import drivest
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

def run_eda_app():
    st.subheader('EDA 화면입니다.')

    diabetes_df = pd.read_csv('data/diabetes.csv')
    
    radio_menu = ['데이터프레임', '통계치']
    selected_radio = st.radio('선택하세요', radio_menu)
    
    if selected_radio == '데이터프레임':
        st.dataframe(diabetes_df)
    
    elif selected_radio == '통계치' :
        st.dataframe( diabetes_df.describe())
    
    columns = diabetes_df.columns
    columns = list(columns)

    selected_columns= st.multiselect('컬럼을 선택하시오.', columns)
    st.dataframe(diabetes_df[selected_columns])

    # 상관계수를 화면에 보여주도록 만듭니다. 

    if st.button("Heatmap보기"):
        fig2 = plt.figure()
        sns.heatmap(diabetes_df.corr(), annot=True , vmax=1 , vmin = -1 , square=2)
        st.pyplot(fig2)

    corr_columns = diabetes_df.columns[diabetes_df.dtypes != object]
    selected_columns = st.multiselect('상관계수 컬럼 선택', corr_columns)
    if len(selected_columns) ==0 :
        st.write("상관계수를 선택해주세요(두 개 선택 권장)")
    else : 
        pass
    

    if len(selected_columns) > 0:
        st.dataframe( diabetes_df[selected_columns].corr())
        fig1 = plt.figure()
        fig = sns.pairplot(data = diabetes_df[selected_columns])
        st.pyplot(fig)
    else :
        st.write('컬럼을 선택하세요')
    

    
    

    # fig2 = plt.figure()
    # fig2 = sns.heatmap(diabetes_df.corr(), annot=True , vmax=1 , vmin = -1 , square=2)
    # st.pyplot(fig2)
    
    # min_max_select = st.selectbox('최대 최소 확인', corr_columns)
    # st.write(min_max_select+"의 최대값")
    # st.dataframe(  diabetes_df.loc[diabetes_df[min_max_select] == diabetes_df[min_max_select].max() , ]  )
    # st.write(min_max_select+"의 최소값")
    # st.dataframe(  diabetes_df.loc[diabetes_df[min_max_select] == diabetes_df[min_max_select].min() , ]  )


    # diabetes_df["Customer Name"].str.contaims('coustomer_name') == True
    # st.dataframe(  diabetes_df.loc[diabetes_df["Customer Name"] == coustomer_name , ]  )

    # coustomer_name = st.text_input("이름을 입력하세요!")
    # st.dataframe(  diabetes_df.loc[diabetes_df["Customer Name"].str.contains(coustomer_name, case=False) == True , ]  )
    