import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictor GU | UNACEM", page_icon="🏭", layout="wide")

col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    try:
        st.image("unacem_logo.png", width=150)
    except:
        pass
with col_titulo:
    st.title("Sistema Predictivo de Calidad Industrial")
    st.markdown("### Cemento Tipo GU - Resistencia Temprana (24h)")
    st.caption("Desarrollado por: **Marco Martínez** | *Analista de Control de Calidad*")

st.divider()

@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_M35_1dia_produccion.pkl')

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("Error de carga. Asegúrese de que el archivo .pkl esté en el repositorio.")
    st.stop()

st.markdown("#### Parámetros Fisicoquímicos y Mineralógicos de Entrada")
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Química (%)**")
        caliza = st.number_input("%Caliza", value=21.5, step=0.1)
        yeso_pct = st.number_input("%Yeso", value=4.74, step=0.1)
        so3 = st.number_input("SO3 (%)", value=2.86, step=0.1)
        alcali = st.number_input("Álcali Eq.", value=0.77, step=0.01)

    with col2:
        st.markdown("**Física de Molienda**")
        m325 = st.number_input("M325 (%)", value=2.95, step=0.1)
        blaine = st.number_input("Blaine (cm²/g)", value=3760.0, step=10.0)
        agua_cem = st.number_input("Agua/Cemento", value=49.0, step=0.1)
        fluidez = st.number_input("Fluidez", value=110.0, step=1.0)

    with col3:
        st.markdown("**DRX Principal**")
        alita = st.number_input("Alita", value=46.2, step=0.5)
        belita = st.number_input("Belita", value=7.0, step=0.5)
        cal_libre = st.number_input("Cal Libre", value=0.24, step=0.01)
        c3a_cub = st.number_input("C3A Cúbico", value=6.24, step=0.1)

    with col4:
        st.markdown("**DRX Yeso y Aluminatos**")
        c3a_ort = st.number_input("C3A Ortorrómbico", value=0.0, step=0.1)
        gypsum = st.number_input("Gypsum", value=2.92, step=0.1)
        hemidrato = st.number_input("Hemidrato", value=0.78, step=0.1)
        anhidrita = st.number_input("Anhidrita", value=0.05, step=0.01)

st.divider()

if st.button("Ejecutar Modelo Predictivo", type="primary", use_container_width=True):
    datos = pd.DataFrame([[caliza, yeso_pct, m325, blaine, so3, alcali, alita, belita, 
                           cal_libre, c3a_cub, c3a_ort, gypsum, hemidrato, anhidrita, agua_cem, fluidez]], 
                         columns=['%Caliza', '%Yeso', 'M325', 'Blaine', 'SO3', 'Alcali_Eq', 'Alita', 
                                  'Belita', 'Cal_Libre', 'C3A_Cúbico', 'C3A_Ortorómbico', 'Gypsum', 
                                  'Hemidrato', 'Anhidrita', 'Agua_Cemento', 'Fluidez'])
    
    with st.spinner("Procesando Tensores..."):
        prediccion = modelo.predict(datos)[0]
    
    st.success("Análisis Predictivo Completado Exitosamente")
    
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #005A9C;">
            <h3 style="color: #333; margin: 0;">Resistencia a la Compresión Estimada (24h)</h3>
            <h1 style="color: #005A9C; font-size: 3em; margin: 0;">{prediccion:.2f} kg/cm²</h1>
            <p style="color: #666; margin: 0;">Precisión (MAE): ± 7.69 kg/cm²</p>
        </div>
    """, unsafe_allow_html=True)
