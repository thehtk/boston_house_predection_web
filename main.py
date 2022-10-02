import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")
df =pd.read_csv("boston.csv")
X=df[['nox','rm','dis','ptratio','lstat']]
y=df['medv']

import sklearn.linear_model.LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)

import streamlit as st
import pickle
import numpy as np
import os
print(os.getcwd())
import time
model=pickle.load(open('model.pkl','rb'))


def predict_forest(nox,rm,dis,ptratio,lstat):
    input=np.array([[nox,rm,dis,ptratio,lstat]]).astype(np.float64)
    prediction=mregressor.predict(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Boston House Predection </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    nox = st.text_input("Nox","Type Here")
    rm = st.text_input("Rm","Type Here")
    dis = st.text_input("Dis","Type Here")
    ptratio = st.text_input("Ptratio","Type Here")
    lstat = st.text_input("Lstat","Type Here")
    


    if st.button("Predict"):
        output=predict_forest(nox,rm,dis,ptratio,lstat)
        st.success('The probability of fire taking place is {}'.format(output))

        

if __name__=='__main__':
    main()
    time.sleep(30)
