import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import LocateControl
import openmeteo_requests
import requests_cache
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import os
import os
os.system("pip uninstall -y pillow")
os.system("pip install --no-cache-dir pillow==9.5.0 --only-binary :all:")
# Install dependensi sistem yang diperlukan
os.system("apt-get update && apt-get install -y zlib1g-dev libjpeg-dev")

# Set page config with improved layout
st.set_page_config(
    page_title="Prediksi Cuaca Indonesia",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a more modern and clean design
st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background-color: #F7F7F7; /* Very light gray background for overall softness */
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        
        /* Header Styles */
        .header-container {
            background: linear-gradient(135deg, #1e3a5f 0%, #4b7c91 100%); /* Navy to light blue gradient */
            color: white;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12); /* Softer shadow */
            margin: 1rem 0 2rem 0;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }
        .header-container h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .header-container p {
            font-size: 1.1rem;
            opacity: 0.8;
        }
        
        /* Card Styles */
        .card {
            background: rgba(255, 255, 255, 0.95); /* Light white with transparency */
            backdrop-filter: blur(5px);
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08); /* Light and soft shadow */
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.12); /* Slight increase in shadow for hover */
        }
        
        /* Button Styles */
        .stButton button {
            background: linear-gradient(135deg, #1e3a5f 0%, #4b7c91 100%) !important; /* Navy to soft blue gradient */
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
        
        /* Metric Styles */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #4b7c91 !important; /* Soft navy blue */
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: #7f8c8d !important; /* Lighter gray for labels */
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1.25rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
        }
        
        /* Radio Button Styles */
        .stRadio > div {
            background: #f8f9fa; /* Very light gray background for radio button container */
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
        
        /* Weather Prediction Section */
        .weather-prediction {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        }
        .weather-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            color: #f39c12; /* Soft yellow for weather icons */
        }
        .weather-temp {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        /* Selectbox Styling - Perbaikan untuk label */
        .stSelectbox [data-testid="stMarkdownContainer"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        .stSelectbox > label {
            display: block !important;
            visibility: visible !important;
            opacity: 


""", unsafe_allow_html=True)


# Rest of the code remains the same as in the original script, just keeping the core functionality
# Load ML model dan scaler
@st.cache_resource
def load_model():
    model = load("model_fixbgt.pkl")
    scaler = load("scaler_fixbgt.pkl")
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

# Card for input method
st.markdown("<h3>📍 Pilih Metode Input Lokasi</h3>", unsafe_allow_html=True)

input_method = st.radio(
    "**Pilih metode:**",  
    ["Pilih Provinsi", "Klik di Peta"],
    index=0,
    horizontal=True,
    label_visibility="visible"  # Pastikan label terlihat
)

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
col1, col2 = st.columns([1.5, 3])

with col2:
    # Create map with colorful style
    m = folium.Map(
        location=st.session_state.zoom_location,
        zoom_start=st.session_state.zoom_level,
        tiles="OpenStreetMap",
        control_scale=True
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

    # Display map
    tmap = st_folium(
        m,
        width=1000,
        height=600,
        returned_objects=["last_clicked"]
    )

with col1:
    if st.session_state.location:
        lat, lon = st.session_state.location
        
        # Display coordinates
        col_lat, col_lon = st.columns(2)
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
                    
                    # Display weather info
                    st.markdown("""
                        <div style='background-color: #e9ecef; padding: 1rem; border-radius: 10px; margin-top: 1rem; text-align: center;'>
                            <h3 style='color: #343a40; margin-bottom: 0.5rem;'>🌤️ Prediksi Cuaca Besok</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display prediction with icon
                    weather_icons = {
                        "Cerah": "☀️",
                        "Cerah Berawan": "🌤️",
                        "Mendung": "☁️",
                        "Hujan Ringan": "🌦️",
                        "Hujan Lebat": "🌧️",
                        "Badai": "⛈️"
                    }
                    
                    st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
                            <h2 style='color: #495057;'>{weather_icons.get(prediction, "🌈")} {prediction}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display weather parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🌡️ Suhu", f"{weather_data['Tavg']:.1f}°C")
                        st.metric("💨 Kecepatan Angin", f"{weather_data['ff_avg']:.1f} m/s")
                    with col2:
                        st.metric("💧 Kelembaban", f"{weather_data['RH_avg']:.0f}%")
                        st.metric("☀️ Durasi Sinar Matahari", f"{weather_data['ss']:.1f} jam")
                    
    else:
        st.warning("⚠️ Pilih lokasi terlebih dahulu untuk melihat prediksi cuaca")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Handle map clicks
if input_method == "Klik di Peta" and tmap and tmap["last_clicked"]:
    lat, lon = tmap["last_clicked"]["lat"], tmap["last_clicked"]["lng"]
    st.session_state.location = [lat, lon]
    st.session_state.zoom_location = [lat, lon]
    st.rerun()
