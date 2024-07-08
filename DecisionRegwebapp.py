import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source
from sklearn import tree
import pickle

st.title("Decision Tree Regressor - Testing")


uploaded_model = st.file_uploader("Choose a model file", type=["pkl"])

if uploaded_model is not None:
    reg = pickle.load(uploaded_model)


    uploaded_file = st.file_uploader("Choose a data file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader('Rows and columns')
        st.write(data.shape)


        X_test = data.iloc[:, :-1].values
        y_test = data.iloc[:, -1].values
        st.write(X_test.shape, y_test.shape)

       
        y_test_pred = reg.predict(X_test)

   
        mse_test = mean_squared_error(y_test, y_test_pred)

        st.subheader("Mean Squared Error for Decision Tree Regressor on Testing Data: " + str(round(mse_test, 2)))

    
        tree = export_graphviz(reg, feature_names=data.columns[:-1])

     
        st.graphviz_chart(tree)
st.title("Download Dataset")
link = '[Mobile Price Prediction](https://github.com/baradgit/DecisiontreeFinal/blob/main/Mobile_Price_Prediction_train%20-%20Copy.csv)'
st.markdown(link, unsafe_allow_html=True)
st.success('Train the model before Test')
if st.button('Train the model'):
    st.markdown(
        """
        <a href="https://appprojectdecisiontree-5eewu4todsnjytxevrctxi.streamlit.app/" 
           style="text-decoration: none;">
            <button style="
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                padding: 15px 32px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 16px; 
                margin: 4px 2px; 
                cursor: pointer;">
                <b>Decision Tree Regressor</b>
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
