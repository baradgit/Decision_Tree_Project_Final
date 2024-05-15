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

# Upload the trained model
uploaded_model = st.file_uploader("Choose a model file", type=["pkl"])

if uploaded_model is not None:
    reg = pickle.load(uploaded_model)

    # Upload the testing dataset
    uploaded_file = st.file_uploader("Choose a data file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader('Rows and columns')
        st.write(data.shape)

        # Assume that the last column is the target variable and the rest are features
        X_test = data.iloc[:, :-1].values
        y_test = data.iloc[:, -1].values
        st.write(X_test.shape, y_test.shape)

        # Predict on testing data
        y_test_pred = reg.predict(X_test)

        # Calculate Mean Squared Error on testing data
        mse_test = mean_squared_error(y_test, y_test_pred)

        st.subheader("Mean Squared Error for Decision Tree Regressor on Testing Data: " + str(round(mse_test, 2)))

        # Export the decision tree to a tree structure
        tree = export_graphviz(reg, feature_names=data.columns[:-1])

        # Display the tree structure using Streamlit's graphviz_chart
        st.graphviz_chart(tree)
