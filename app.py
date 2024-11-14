import streamlit as st
import pandas as pd
import joblib
import pickle
import requests
from sklearn.preprocessing import RobustScaler

model = joblib.load("assets/model/model_klasifikasi_awal.pkl") 
scaler = joblib.load("assets/scaler/scaler_awal.pkl")

df = pd.read_csv("assets/data/rumah_sakit.csv")

def knn_recommendation(input_lat, input_lon, data, k=10):
    with open('assets/model/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    
    distances, indices = knn_model.kneighbors([[input_lat, input_lon]])
    nearest_locations = data.iloc[indices[0]]
    nearest_locations = nearest_locations.assign(Distance=distances[0])

    columns_to_return = ['Nama Dokter', 'Jenis Dokter', 'Rumah Sakit', 'Alamat', 'Distance']
    columns_to_return = [col for col in columns_to_return if col in nearest_locations.columns]

    return nearest_locations[columns_to_return]

def get_coordinates_from_ip():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data['status'] == 'success':
            return data['lat'], data['lon']
    except:
        st.write("Error retrieving location data.")
    return None, None

st.title("Aplikasi Prediksi Kesehatan dan Rekomendasi Rumah Sakit")
st.write("Masukkan data untuk prediksi kondisi kesehatan Anda dan rekomendasi rumah sakit yang sesuai.")

st.sidebar.image("assets/img/logo.png", width=200)

with st.form("prediction_form"):
    st.write("Masukkan fitur untuk prediksi:")
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", options=[0, 1])  
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=0)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])  
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, format="%.1f")

    age_thalach = age * thalach
    cp_oldpeak = cp * oldpeak

    submit = st.form_submit_button("Submit")

    if submit:
        input_data = pd.DataFrame([[age, sex, cp, thalach, exang, oldpeak, age_thalach, cp_oldpeak]], 
                                  columns=["age", "sex", "cp", "thalach", "exang", "oldpeak", "age_thalach", "cp_oldpeak"])

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        st.write(f"Hasil Prediksi: {'Positive' if prediction != 0 else 'Negative'}")

        if prediction != 0:
            user_lat, user_lon = get_coordinates_from_ip()
            
            if user_lat and user_lon:
                rekomendasi = knn_recommendation(user_lat, user_lon, df)
                st.write("Rekomendasi Rumah Sakit Terdekat:")
                st.dataframe(rekomendasi)
            else:
                st.write("Tidak dapat mendapatkan lokasi otomatis. Coba izinkan akses lokasi di pengaturan browser atau masukkan secara manual.")
        else:
            st.write("Hasil prediksi menunjukkan kemungkinan kondisi yang tidak normal. Segera periksakan ke puskesmas atau fasilitas kesehatan terdekat untuk pemeriksaan lebih lanjut.")
