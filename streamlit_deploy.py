import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('gbr_model.joblib')

st.title('Prediksi Model Machine Learning')

# Buat form input untuk user (sesuaikan dengan fitur dataset Anda)
fitur_1 = st.number_input('Input Fitur 1')
fitur_2 = st.number_input('Input Fitur 2')

if st.button('Prediksi'):
    # Format input menjadi DataFrame agar sesuai dengan format saat training
    input_data = pd.DataFrame({'NamaKolom1': [fitur_1], 'NamaKolom2': [fitur_2]})
    
    # Eksekusi prediksi
    hasil = model.predict(input_data)
    st.success(f'Hasil Prediksi: {hasil[0]}')