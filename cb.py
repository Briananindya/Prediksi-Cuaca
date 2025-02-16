import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import LocateControl
import requests_cache
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import openmeteo_requests
import openmeteo_sdk
import gzip
import pickle
import gdown
import requests

# Set page config with improved layout
st.set_page_config(
    page_title="Prediksi Cuaca Indonesia",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS 
st.markdown("""
    <style>
        /* Global Styles /
        .stApp {
            background-color: #F7F7F7; / Very light gray background for overall softness */
            font-family: 'Inter', 'Roboto', sans-serif;
        }
    
        /* Header Styles /
        .header-container {
            background: inherit; / Use default color to adapt to theme /
            color: inherit; / Use default color to adapt to theme /
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12); / Softer shadow /
            margin: 1rem 0 2rem 0;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }
        .header-container h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            color: inherit; / Use default color to adapt to theme /
        }
        .header-container p {
            font-size: 1.1rem;
            opacity: 0.8;
            color: inherit; / Use default color to adapt to theme */
        }
        /* Card Styles /
        .card {
            background: inherit; / Default - uses theming /
            backdrop-filter: blur(5px);
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08); / Light and soft shadow /
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.12); / Slight increase in shadow for hover */
        }

        /* Button Styles /
        .stButton button {
            background: linear-gradient(135deg, #1e3a5f 0%, #4b7c91 100%) !important; / Navy to soft blue gradient */
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.875rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 5px 15px rgba(30,58,95,0.3) !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(30,58,95,0.4) !important;
        }
    
        /* Metric Styles /
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: inherit; / Use default color to adapt to theme /
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: inherit; / Use default color to adapt to theme /
        }
        .metric-card {
            background: inherit; / Default - uses theming */   
            border-radius: 15px;
            padding: 1.25rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
        }
    
        /* Radio Button Styles /
        .stRadio > div {
            background: inherit; / Default - uses theming */
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .stRadio [data-testid="stMarkdownContainer"] > div {
            gap: 1.5rem !important;
        }

        /* Map Container */
        [data-testid="stIframe"] {
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        /* Weather Prediction Section /
        .weather-prediction {
            background: inherit; / Allow theme to show default colors /
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
            color: inherit; / Use default color to adapt to theme /
        }
        .weather-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            color: #f39c12; / Soft yellow for weather icons /
        }
        .weather-temp {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50; / Navy blue for temperature */
        }

        /* Selectbox Styling - Perbaikan untuk label /
        .stSelectbox [data-testid="stMarkdownContainer"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        .stSelectbox > label {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            color: inherit; / Use default color to adapt to theme */

}""", unsafe_allow_html=True)


MODEL_PATH = "model_fixbgtoke.pkl"
SCALER_PATH = "scaler_fixbgtoke.pkl"

MODEL_URL_ID = "1rQgSl9pKhzwUJxOtk2ToZ2g7XL7ldr0V"
SCALER_URL_ID = "1JGqPcTpH-QUtpnMR_YsnEidXkDU6UG1F"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Mengunduh model dari Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_URL_ID}&confirm=t", MODEL_PATH, quiet=False)

    if not os.path.exists(SCALER_PATH):
        st.warning("Mengunduh scaler dari Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={SCALER_URL_ID}&confirm=t", SCALER_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("File model atau scaler tidak ditemukan! Unduh secara manual.")
        st.stop()

    st.success("Model dan Scaler berhasil diunduh dan dimuat!")
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)

    return model, scaler

model, scaler = load_model()


# Fungsi untuk mendapatkan data cuaca
def fetch_weather_data(latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    openmeteo = openmeteo_requests.Client(session=cache_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2025-02-09",
        "end_date": "2025-02-11",
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean","wind_speed_10m_max", "wind_direction_10m_dominant", "sunshine_duration"],
        "wind_speed_unit": "ms"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        daily = responses[0].Daily()

        weather_dict = {
            "Tavg": float(daily.Variables(0).ValuesAsNumpy()[0]),
            "RH_avg": float(daily.Variables(1).ValuesAsNumpy()[0]),
            "ff_avg": float(daily.Variables(2).ValuesAsNumpy()[0]),
            "ddd_x": float(daily.Variables(3).ValuesAsNumpy()[0]),
            "ss": float(daily.Variables(4).ValuesAsNumpy()[0])/3600,
            "RR": 0,
            "RR_3days_avg": 0,
            "ddd_x_yesterday": float(daily.Variables(3).ValuesAsNumpy()[0]),
            "ff_avg_yesterday": float(daily.Variables(2).ValuesAsNumpy()[0]),
            "RR_yesterday": 0,
            "Tavg_yesterday": float(daily.Variables(0).ValuesAsNumpy()[0]),
            "RH_avg_yesterday": float(daily.Variables(1).ValuesAsNumpy()[0]),
        }
        return weather_dict
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

# Load provinsi_data
provinsi_data = {   
    "Bali": [-8.75, 115.17],
    "Banten": [-6.26151, 106.75084],
    "Bengkulu": [-3.8582, 102.3367],
    "DI Yogyakarta": [-7.731, 110.354],
    "DKI Jakarta": [-6.10781, 106.88053],
    "Gorontalo": [0.6385, 122.8525],
    "Jambi": [-1.6019, 103.48444],
    "Jawa Barat": [-6.7, 106.85],
    "Jawa Tengah": [-6.86817, 109.12103],
    "Jawa Timur": [-5.8511, 112.6578],
    "Kalimantan Barat": [1.74, 109.3],
    "Kalimantan Selatan": [-3.442, 114.754],
    "Kalimantan Tengah": [-0.56, 114.53],
    "Kalimantan Timur": [2.14562, 117.43375],
    "Kalimantan Utara": [4.13, 117.67],
    "Kep. Bangka Belitung": [-2.17, 106.13],
    "Kep. Riau": [1.11667, 104.11667],
    "Lampung": [-5.17236, 105.18],
    "Maluku": [-3.25, 127.08],
    "Maluku Utara": [1.84092, 127.7905],
    "Nanggroe Aceh Darussalam": [5.87655, 95.33785],
    "Nusa Tenggara Barat": [-8.75277, 116.24982],
    "Nusa Tenggara Timur": [-8.48673, 119.88683],
    "Papua": [-1.19069, 136.10361],
    "Papua Barat": [-0.89118, 131.28575],
    "Riau": [0.407, 101.217],
    "Sulawesi Barat": [-3.55074, 118.98054],
    "Sulawesi Selatan": [-3.04524, 119.81885],
    "Sulawesi Tengah": [1.12114, 120.79433],
    "Sulawesi Tenggara": [-4.18059, 121.61077],
    "Sulawesi Utara": [3.68594, 125.52881],
    "Sumatera Barat": [-0.99639, 100.37222],
    "Sumatera Selatan": [-2.89468, 104.70129],
    "Sumatera Utara": [3.62114, 98.71485],
}

# Header 
st.markdown("""
    <div class="header-container">
        <h1>🌦️ Prediksi Cuaca Indonesia</h1>
        <p>Kelompok 3 | Analisis Cuaca Terkini</p>
    </div>
""", unsafe_allow_html=True)

# Inisialisasi session state
if "location" not in st.session_state:
    st.session_state.location = None
if "zoom_location" not in st.session_state:
    st.session_state.zoom_location = [-2.5, 118.0]
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 5
if "weather_prediction" not in st.session_state:
    st.session_state.weather_prediction = None


# Create map with colorful style
m = folium.Map(
    location=st.session_state.zoom_location,
    zoom_start=st.session_state.zoom_level,
    tiles="OpenStreetMap",
    control_scale=True
)

# Card for input method
st.markdown("<h3>📍 Pilih Metode Input Lokasi</h3>", unsafe_allow_html=True)

input_method = st.radio(
    "**Pilih metode:**",  
    ["Pilih Provinsi", "Klik di Peta"],
    index=0,
    horizontal=True,
    label_visibility="visible"  
)

    # Add location control
LocateControl(auto_start=False).add_to(m)
    
    # Add marker if location is selected
if st.session_state.location:
        lat, lon = st.session_state.location
        folium.Marker(
            [lat, lon],
            popup=f"Lat: {lat:.4f}, Lon: {lon:.4f}",
            icon=folium.Icon(color="red", icon="cloud"),
            draggable=False
        ).add_to(m)
# Bungkus peta dalam div agar bisa dikontrol dengan CSS
with st.container():
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    tmap = st_folium(
        m,
        width="100%",  # Biarkan width otomatis
        height=600,  # Nilai default, tapi akan ditimpa oleh CSS
        returned_objects=["last_clicked"]
    )
    st.markdown('</div>', unsafe_allow_html=True)


# Handle map clicks
if input_method == "Klik di Peta" and tmap and isinstance(tmap.get("last_clicked"), dict):
    last_clicked = tmap["last_clicked"]
    
    if last_clicked:  # Pastikan last_clicked tidak None
        lat, lon = last_clicked["lat"], last_clicked["lng"]
        
        # Update session state hanya jika lokasi berubah
        if "location" not in st.session_state or st.session_state.location != [lat, lon]:  
            st.session_state.location = [lat, lon]
            st.session_state.zoom_location = [lat, lon]
            st.experimental_rerun()


if input_method == "Pilih Provinsi":
    # Gunakan tampilan yang lebih simpel
    st.markdown("**Pilih Provinsi:**", unsafe_allow_html=True)
    
    target_provinsi = st.selectbox(
        "",  # Empty label
        ["- Pilih Provinsi -"] + sorted(list(provinsi_data.keys())),
        help="Pilih provinsi untuk melihat prediksi cuaca",
        key="provinsi_selector",
        label_visibility="collapsed"
    )
    
    # Update location based on selection
    if target_provinsi == "- Pilih Provinsi -":
        st.session_state.location = None
        st.session_state.zoom_location = [-2.5, 118.0]
        st.session_state.zoom_level = 5
    else:
        st.session_state.location = provinsi_data[target_provinsi]
        st.session_state.zoom_location = provinsi_data[target_provinsi]
        st.session_state.zoom_level = 8
else:
    st.info("🎯 Klik pada peta untuk memilih lokasi yang diinginkan")


st.markdown('</div>', unsafe_allow_html=True)

# Layout columns
st.markdown(
    """
    <style>
        .map-container iframe {
            width: 100% !important;
            height: 80vh !important;  /* Sesuaikan tinggi dengan viewport */
            border-radius: 15px;  /* Opsional: Tambahkan border radius */
        }
    </style>
    """,
    unsafe_allow_html=True
)


if st.session_state.location:
        lat, lon = st.session_state.location
        
        # Display coordinates
        col_lat, col_lon = st.columns([3,2])
        with col_lat:
            st.metric("📍 Latitude", f"{lat:.4f}")
        with col_lon:
            st.metric("📍 Longitude", f"{lon:.4f}")
        
        # Weather prediction button
        if st.button("🔮 Prediksi Cuaca", use_container_width=True):
            with st.spinner("Mengambil data cuaca..."):
                # Get weather data
                weather_data = fetch_weather_data(lat, lon)
                
                if weather_data:
                    # Prepare features
                    features = ['RR', 'RR_yesterday', 'RR_3days_avg', 'Tavg', 'Tavg_yesterday',
                                'RH_avg', 'RH_avg_yesterday', 'ff_avg', 'ff_avg_yesterday',
                                'ddd_x', 'ddd_x_yesterday', 'ss']
                    
                    df_live = pd.DataFrame([weather_data])
                    df_live = df_live[features]
                    
                    # Scale features
                    df_live_scaled = scaler.transform(df_live)
                    
                    # Make prediction
                    prediction = model.predict(df_live_scaled)[0]
                    st.session_state.weather_prediction = prediction
                    
                    # Display prediction with icon
                    weather_icons = {
                        "Cerah": "☀️",
                        "Cerah Berawan": "🌤️",
                        "Mendung": "☁️",
                        "Hujan Ringan": "🌦️",
                        "Hujan Lebat": "🌧️",
                        "Badai": "⛈️"
                    }
                    with st.container():
                        st.info("**Informasi Cuaca dan Prediksi Besok**")
                    # Buat layout dengan 2 kolom yang ukurannya sama
                    col1, col2 = st.columns(2)
    
                    # Kolom 1: Menampilkan 4 variabel utama
                    with col1:
                        st.metric("🌡️ Suhu", f"{weather_data['Tavg']:.1f}°C")
                        st.metric("💨 Kecepatan Angin", f"{weather_data['ff_avg']:.1f} m/s")
                        st.metric("💧 Kelembaban", f"{weather_data['RH_avg']:.0f}%")
                        st.metric("☀️ Durasi Sinar Matahari", f"{weather_data['ss']:.1f} jam")

                    # Kolom 2: Menampilkan Prediksi Cuaca dalam Ukuran Besar, di tengah kolom 1
                    with col2:
                        st.markdown("""
                            <style>
                                .prediksi-container {
                                    display: flex;
                                    flex-direction: column;
                                    align-items: center;
                                    height: 100%;
                                    padding: 3rem;
                                    background-color: inherit;
                                    border-radius: 15px;
                                    text-align: center;
                                }
                                .prediksi-title {
                                    font-size: 1rem;
                                    margin-bottom: 1rem;
                                }
                                .prediksi-text {
                                    font-size: 2.5rem;
                                    font-weight: bold;
                                }
                            </style>
                            <div class="prediksi-container">
                                <h6 class="prediksi-title">🌤️ Prediksi Cuaca Besok</h6>
                                <h1 class="prediksi-text">{weather_icon} {prediction}</h1>
                            </div>
                        """.replace("{weather_icon}", weather_icons.get(prediction, "🌈")).replace("{prediction}", prediction), unsafe_allow_html=True)
                    
else:
    st.warning("⚠️ Pilih lokasi terlebih dahulu untuk melihat prediksi cuaca")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fungsi untuk menampilkan halaman "Tentang Kami"
def about_us():
    # Judul utama dengan jarak tambahan
    st.markdown("<h3>✨ Tentang Kami ✨</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)  # Spasi tambahan

    # List anggota tim dengan emoji sesuai cuaca
    developers = [
        {"name": "Anak Agung Naraya Putra", "nim": "103012300328", "image": "gungwah.jpg", "emoji": "🌤️"},
        {"name": "Nabila Putri Azhari", "nim": "103012300316", "image": "nabila.jpg", "emoji": "🌧️"},
        {"name": "Reevanza Abel Desta Arifin", "nim": "103012330104", "image": "reevanza.jpg", "emoji": "⛈️"},
        {"name": "Brian Anindya", "nim": "103012300463", "image": "brian.jpg", "emoji": "🌪️"},
    ]

    # Membuat tampilan dalam 2 kolom (2x2 layout)
    col1, col2 = st.columns(2)

    for index, dev in enumerate(developers):
        with (col1 if index % 2 == 0 else col2):  # Alternatif ke kolom kiri & kanan
            with st.container():  # Kotak pemisah untuk tiap anggota
                img_col, text_col = st.columns([1, 2])  # Foto di kiri, teks di kanan
                
                with img_col:
                    # Cek apakah gambar tersedia sebelum menampilkannya
                    if os.path.exists(dev["image"]):
                        st.image(dev["image"], width=230)
                    else:
                        st.warning(f"Gambar tidak ditemukan: {dev['image']}")  # Debugging error

                with text_col:
                    st.markdown(f"<h4>{dev['emoji']} {dev['name']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: {30};'><strong>NIM:</strong> {dev['nim']}</p>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)  # Garis pemisah antar anggota

    # Informasi tambahan di bawah
    st.markdown("### 📌 Informasi Tim")
    st.write(
        "Kami adalah tim yang berdedikasi untuk mengembangkan aplikasi cuaca berbasis AI. "
        "Dengan semangat kolaborasi, kami terus berinovasi untuk memberikan prediksi cuaca yang akurat dan bermanfaat bagi masyarakat."
    )
    st.write("")  # Jarak tambahan

# Tombol untuk membuka halaman "Tentang Kami"
if st.button("📌 Tentang Kami"):
    about_us()
