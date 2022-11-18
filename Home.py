import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.image('./pic/welcome.jpg')

html_8="""
<div style="background-color:#66CCCC;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:Bisque">
<center><h5>การทำนายสภาพอากาศ</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/cirhh.csv")
st.write(dt.head(10))

data1 = dt['Status'].sum()
data2 = dt['Hepatomegaly'].sum()
data3 = dt['Edema'].sum()



dx=[data1,data2,data3]
dx2=pd.DataFrame(dx, index=["d1", "d2", "d3"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.area_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8="""
<div style="background-color:#66CCCC;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:Bisque">
<center><h5>การทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

Status=st.number_input("กรุณาเลือกข้อมูล Status")
Hepatomegaly=st.number_input("กรุณาเลือกข้อมูล Hepatomegaly")
Edema=st.number_input("กรุณาเลือกข้อมูล Edema")



if st.button("ทำนายผล"):
   loaded_model = pickle.load(open('./data/weather_model.sav','rb'))
   input_data =  (Status,Hepatomegaly,Edema)
   # changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)
   # reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   prediction = loaded_model.predict(input_data_reshaped)
   st.write(prediction)
   if prediction == 'low':
        st.image('./pic/usa.jpg')
   elif prediction == 'medium':
        st.image('./pic/eu.jpg')
   else:
        st.image('./pic/uk.jpg')
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")



