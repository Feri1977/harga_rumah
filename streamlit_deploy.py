import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('gbr_model.joblib')

st.title('Prediksi Model Machine Learning')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Mengambil data dari request JSON
    prediction = model.predict(data)  # Melakukan prediksi (harus dalam bentuk 2D array)
    return jsonify({'prediction': prediction.tolist()})
 
if __name__ == '__main__':
    app.run(debug=True)