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
<center><h5>การทำนายข้อมูลคุณภาพนม</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/football.csv")
st.write(dt.head(10))

data1 = dt['size'].sum()
data2 = dt['cm'].sum()




dx=[data1,data2]
dx2=pd.DataFrame(dx, index=["d1", "d2"])

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

size=st.number_input("กรุณาเลือกข้อมูล size")
cm=st.slider("กรุณาเลือกข้อมูล cm")


if st.button("ทำนายผล"):
   loaded_model = pickle.load(open('./data/ball_model.sav', 'rb'))
   input_data =  (size,cm)
   # changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)
   # reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   prediction = loaded_model.predict(input_data_reshaped)
   st.write(prediction)
   if prediction == 'low':
        st.image('./pic/virginica.jpg')
   elif prediction == 'medium':
        st.image('./pic/versicolor.jpg')
   else:
        st.image('./pic/setosa.jpg')
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

