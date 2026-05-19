import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de página
st.set_page_config(page_title="Predicción de Viviendas", layout="centered")

# Cargar el modelo y el escalador desde la carpeta 'modelos'
@st.cache_resource
def load_models():
    modelo = joblib.load('modelos/modelo_rf.pkl')
    scaler = joblib.load('modelos/scaler.pkl')
    return modelo, scaler

modelo_rf, scaler = load_models()

# Interfaz web
st.title("🏡 Predicción de Precios de Viviendas en California")
st.write("Ingrese los datos de las características de la vivienda para predecir su valor estimado.")

# Inputs del usuario organizados en columnas
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input('Longitud', value=-122.23)
    latitude = st.number_input('Latitud', value=37.88)
    housing_median_age = st.number_input('Edad Media de la Vivienda', value=41.0)
    total_rooms = st.number_input('Total de Habitaciones', value=880.0)

with col2:
    total_bedrooms = st.number_input('Total de Dormitorios', value=129.0)
    population = st.number_input('Población', value=322.0)
    households = st.number_input('Cantidad de Hogares', value=126.0)
    median_income = st.number_input('Ingreso Medio (x $10,000)', value=8.3252)

ocean_proximity = st.selectbox('Proximidad al Océano', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Creación del dataframe con los inputs para predecir
if st.button('🎯 Calcular Predicción'):
    # Adaptar los inputs a la misma estructura de columnas del entrenamiento
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity_INLAND': [1 if ocean_proximity == 'INLAND' else 0],
        'ocean_proximity_ISLAND': [1 if ocean_proximity == 'ISLAND' else 0],
        'ocean_proximity_NEAR BAY': [1 if ocean_proximity == 'NEAR BAY' else 0],
        'ocean_proximity_NEAR OCEAN': [1 if ocean_proximity == 'NEAR OCEAN' else 0]
    })
    
    # Escalar y predecir
    input_scaled = scaler.transform(input_data)
    prediccion = modelo_rf.predict(input_scaled)[0]
    
    st.success(f"### El valor estimado de la vivienda es: ${prediccion:,.2f}")

# Requisito de la Rúbrica: Enlace a Colab y Datos ISIL
st.markdown("---")
# REEMPLAZA EL ENLACE Y TUS DATOS AQUÍ
st.markdown("📄 **[Enlace a mi cuaderno de código COLAB](https://colab.research.google.com/drive/1X6ci8GOkI7LS_RH17I22BLc9zIgH_IBG?usp=sharing)**")
st.markdown("👤 **Nombre:** [Leonar Vicencio Barra]")
st.markdown("🎓 **Código ISIL:** [70838952]")