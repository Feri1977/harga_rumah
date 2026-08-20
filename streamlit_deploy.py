import streamlit as st
import joblib
import pandas as pd

# Muat model yang ada di folder yang sama
model = joblib.load('gbr_model.joblib')

st.title('Prediksi Model ML Lokal')

# Sesuaikan dengan nama kolom pada dataset Github Anda
fitur_1 = st.number_input('Input Fitur 1')
fitur_2 = st.number_input('Input Fitur 2')

if st.button('Prediksi'):
    input_data = pd.DataFrame({'Kolom1': [fitur_1], 'Kolom2': [fitur_2]})
    hasil = model.predict(input_data)
    st.success(f'Hasil Prediksi: {hasil[0]}')