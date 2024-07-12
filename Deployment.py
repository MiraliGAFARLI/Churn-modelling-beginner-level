import streamlit as st
import pandas as pd
import numpy as np
# # Title / Text

st.title("Training data aggregates")



# # sidebar

st.sidebar.title("Features")
cr_score=st.sidebar.number_input("Credit Score",350,850, 650)
geog=st.sidebar.radio("Select a country", ( "France", "Germany", "Spain"))
sex=st.sidebar.radio("Gender", ("Male", "Female"))
age=int(st.sidebar.slider("Age", 0, 100, 39, 1))
tnr=st.sidebar.slider("Tenure",0, 20, 5, 1)
blnc=int(st.sidebar.number_input("Balance", value= 0))
nop=st.sidebar.slider("Number of products", 0, 20, 1, 1)
hascrcard=int(st.sidebar.checkbox("Has Credit Card", value= True))
isactv=int(st.sidebar.checkbox("Is active member", value= True))
estslry=int(st.sidebar.number_input("Estimated Salary", value= 0))






df = pd.read_csv("Churn_modelling.csv")

aggregates = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1).describe().T

st.table(aggregates.applymap(lambda x: f'{x:.2f}'))

df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, inplace=True)

from tensorflow.keras.models import load_model

model = load_model("churn_model.h5")

import pickle
scaler = pickle.load(open("scaler_churn", 'rb'))

my_dict = {
           "CreditScore": cr_score,
           "Geography": geog,
           "Gender": sex,
           "Age": age,
           "Tenure": tnr,
           "Balance": blnc,
           "NumOfProducts": nop,
           "HasCrCard": hascrcard,
           "IsActiveMember": isactv,
           "EstimatedSalary": estslry
          }


inputs =pd.DataFrame.from_dict([my_dict])

df_with_inputs = pd.concat([df, inputs])

df_dummified = pd.get_dummies(columns=["Gender", "Geography"], data=df_with_inputs, drop_first= True)
df_dummified[['Gender_Male', 'Geography_Germany', 'Geography_Spain']] = df_dummified[['Gender_Male','Geography_Germany', 'Geography_Spain']].astype(int)

inputs_dummified = df_dummified.tail(1)

inputs_dummified_scaled = scaler.transform(inputs_dummified)



st.write("# Prediction")

st.table(inputs)

if st.button("Predict"):
    pred = model.predict(inputs_dummified_scaled)
    st.success(f"{pred[0][0]:.2%}")
    st.success("Exited" if pred[0][0]>0.5 else "Stayed")