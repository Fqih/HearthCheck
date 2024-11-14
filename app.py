import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import RobustScaler
from geopy.geocoders import Nominatim

# Load model & scaler
model = joblib.load("assets/model/model_klasifikasi.pkl") 
scaler = joblib.load("assets/scaler/KlasifikasiJantung.pkl")

# dataset rumah sakit
df = pd.read_csv("assets/data/rumah_sakit.csv")  # Pastikan Anda memuat data yang benar

# Fungsi untuk rekomendasi rumah sakit menggunakan KNN
def knn_recommendation(input_lat, input_lon, data, k=10):
    with open('assets/model/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    
    distances, indices = knn_model.kneighbors([[input_lat, input_lon]])
    nearest_locations = data.iloc[indices[0]]
    nearest_locations = nearest_locations.assign(Distance=distances[0])

    columns_to_return = ['Nama Dokter', 'Jenis Dokter', 'Rumah Sakit', 'Alamat', 'Distance']
    columns_to_return = [col for col in columns_to_return if col in nearest_locations.columns]

    return nearest_locations[columns_to_return]

# Fungsi untuk dapetin latitude dan longitude
def get_coordinates(address):
    geolocator = Nominatim(user_agent="healthcare_recommendation")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

st.title("Aplikasi Prediksi Kesehatan dan Rekomendasi Rumah Sakit")
st.write("Masukkan data untuk prediksi kondisi kesehatan Anda dan rekomendasi rumah sakit yang sesuai.")

st.sidebar.image("assets/img/logo.png", width=200)

with st.form("prediction_form"):
    st.write("Masukkan fitur untuk prediksi:")
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", options=[0, 1])  # 0 untuk Female, 1 untuk Male
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=0)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])  # 0 untuk tidak, 1 untuk ya
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, format="%.1f")

    age_thalach = age * thalach
    cp_oldpeak = cp * oldpeak

    submit = st.form_submit_button("Submit")

    if submit:
        input_data = pd.DataFrame([[age, sex, cp, thalach, exang, oldpeak, age_thalach, cp_oldpeak]], 
                                  columns=["age", "sex", "cp", "thalach", "exang", "oldpeak", "age_thalach", "cp_oldpeak"])

        input_data_scaled = scaler.transform(input_data)


        prediction = model.predict(input_data_scaled)[0]

        # Menampilkan hasil prediksi
        st.write(f"Hasil Prediksi: {'Positive' if prediction != 0 else 'Negative'}")

        if prediction != 0:
            user_location = st.text_input("Masukkan lokasi atau alamat Anda:")
            if user_location:
                user_lat, user_lon = get_coordinates(user_location)
                
                if user_lat and user_lon:
                    rekomendasi = knn_recommendation(user_lat, user_lon, df)
                    
                    st.write("Rekomendasi Rumah Sakit Terdekat:")
                    st.dataframe(rekomendasi)
                else:
                    st.write("Lokasi tidak ditemukan. Harap masukkan alamat yang valid.")
        else:
            st.write("Hasil prediksi menunjukkan kemungkinan kondisi yang tidak normal. Segera periksakan ke puskesmas atau fasilitas kesehatan terdekat untuk pemeriksaan lebih lanjut.")
