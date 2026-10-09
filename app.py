import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import unicodedata

# Configuración de página
st.set_page_config(
    page_title="QuAI | M.A.R.C.O. - UNACEM",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS avanzados (Compacto y Sofisticado)
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 95% !important;
    }
    .brand-tag {
        font-size: 0.82rem;
        font-weight: 700;
        color: #E30613;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    h1 {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }
    .sub-meta {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #d1d5db !important;
    }
    div[data-testid="stNumberInput"] input {
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        padding: 4px 8px !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #E30613 0%, #B8000C 100%);
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 12px 20px;
        box-shadow: 0 4px 14px rgba(227, 6, 19, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #FF1A27 0%, #D0000E 100%);
        transform: translateY(-1px);
    }
    .result-card {
        background: radial-gradient(circle at top, #1f242e 0%, #11141a 100%);
        border: 1px solid #303642;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .result-value {
        font-size: 3.8rem;
        font-weight: 800;
        color: #f9fafb;
        line-height: 1;
    }
    .result-unit {
        font-size: 1.4rem;
        color: #E30613;
        font-weight: 700;
    }
    .footer-disclaimer {
        text-align: center;
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1f242e;
    }
</style>
""", unsafe_allow_html=True)

# Carga de modelos
@st.cache_resource
def cargar_modelos():
    m1 = None
    m28 = None
    if os.path.exists("modelo_M35_1dia_produccion.pkl"):
        m1 = joblib.load("modelo_M35_1dia_produccion.pkl")
    if os.path.exists("modelo_M34_28d_produccion.pkl"):
        m28 = joblib.load("modelo_M34_28d_produccion.pkl")
    return m1, m28

mod_1d, mod_28d = cargar_modelos()

# Función adaptadora para compatibilidad de variables en XGBoost y LightGBM
def predecir_con_modelo(modelo, df_input):
    expected_cols = None
    
    # Extraer nombres exactos de entrenamiento
    if hasattr(modelo, "feature_names_in_"):
        expected_cols = list(modelo.feature_names_in_)
    elif hasattr(modelo, "get_booster"):
        try:
            expected_cols = modelo.get_booster().feature_names
        except Exception:
            pass
    elif hasattr(modelo, "booster_"):
        try:
            expected_cols = modelo.booster_.feature_name()
        except Exception:
            pass

    if expected_cols:
        def norm(t):
            return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(t)).encode('ASCII', 'ignore').decode('utf-8').lower())

        input_clean = {norm(c): c for c in df_input.columns}
        df_alineado = pd.DataFrame()

        for col in expected_cols:
            col_c = norm(col)
            if col in df_input.columns:
                df_alineado[col] = df_input[col]
            elif col_c in input_clean:
                df_alineado[col] = df_input[input_clean[col_c]]
            else:
                df_alineado[col] = 0.0

        return float(modelo.predict(df_alineado)[0])
    else:
        return float(modelo.predict(df_input)[0])

# Encabezado institucional y marca QuAI
col_logo, col_tit = st.columns([1.8, 5.2])
with col_logo:
    if os.path.exists("unacem_logo.png"):
        st.image("unacem_logo.png", width=220)

with col_tit:
    st.markdown("<div class='brand-tag'>QuAI PRESENTA: M.A.R.C.O.</div>", unsafe_allow_html=True)
    st.markdown("<h1>Sistema Predictivo de Resistencia a la Compresión</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-meta'>División de Control de Calidad Atocongo &nbsp;|&nbsp; <b>by Marco Martínez</b></div>", unsafe_allow_html=True)

# Selectores de configuración
c_tipo, c_horiz, c_mod = st.columns([2.5, 2.5, 2])
with c_tipo:
    tipo_cemento = st.selectbox("Tipo de Cemento", ["Cemento Tipo GU"])
with c_horiz:
    horizonte = st.selectbox("Horizonte de Producción", ["28 Días", "1 Día"])
with c_mod:
    modelo_nombre = "LightGBM (M34)" if "28 Días" in horizonte else "XGBoost (M35)"
    st.text_input("Modelo Activo", value=modelo_nombre, disabled=True)

st.markdown("<hr style='margin: 0.7rem 0; border-color: #262b33;'>", unsafe_allow_html=True)

# Grid de 16 variables de proceso (4x4)
c1, c2, c3, c4 = st.columns(4)

with c1:
    caliza = st.number_input("%Caliza", value=21.50, step=0.10, format="%.2f")
    so3 = st.number_input("SO3 (%)", value=2.86, step=0.05, format="%.2f")
    cal_libre = st.number_input("Cal Libre", value=0.24, step=0.05, format="%.2f")
    hemidrato = st.number_input("Hemidrato", value=0.78, step=0.10, format="%.2f")

with c2:
    yeso = st.number_input("%Yeso", value=4.74, step=0.10, format="%.2f")
    alcali_eq = st.number_input("Álcali_Eq", value=0.77, step=0.01, format="%.2f")
    c3a_cub = st.number_input("C3A Cúbico", value=6.24, step=0.10, format="%.2f")
    anhidrita = st.number_input("Anhidrita", value=0.05, step=0.05, format="%.2f")

with c3:
    m325 = st.number_input("M325 (%)", value=2.95, step=0.10, format="%.2f")
    alita = st.number_input("Alita", value=46.20, step=0.50, format="%.2f")
    c3a_orto = st.number_input("C3A Ortorrómbico", value=0.00, step=0.10, format="%.2f")
    agua_cemento = st.number_input("Agua/Cemento", value=49.00, step=0.50, format="%.2f")

with c4:
    blaine = st.number_input("Blaine (cm²/g)", value=3760.0, step=10.0, format="%.2f")
    belita = st.number_input("Belita", value=7.00, step=0.50, format="%.2f")
    gypsum = st.number_input("Gypsum", value=2.92, step=0.10, format="%.2f")
    fluidez = st.number_input("Fluidez", value=110.00, step=1.00, format="%.2f")

st.write("")

# Inferencia
if st.button("Calcular Predicción", use_container_width=True):
    input_data = pd.DataFrame([{
        '%Caliza': caliza,
        '%Yeso': yeso,
        'M325': m325,
        'Blaine': blaine,
        'SO3': so3,
        'Álcali_Eq': alcali_eq,
        'Alita': alita,
        'Belita': belita,
        'Cal_Libre': cal_libre,
        'C3A_Cúbico': c3a_cub,
        'C3A_Ortorómbico': c3a_orto,
        'Gypsum': gypsum,
        'Hemidrato': hemidrato,
        'Anhidrita': anhidrita,
        'Agua_Cemento': agua_cemento,
        'Fluidez': fluidez
    }])

    if "28 Días" in horizonte:
        if mod_28d is not None:
            pred = predecir_con_modelo(mod_28d, input_data)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-value">{int(round(pred))} <span class="result-unit">kg/cm²</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No se encontró el archivo 'modelo_M34_28d_produccion.pkl'.")
    else:
        if mod_1d is not None:
            pred = predecir_con_modelo(mod_1d, input_data)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-value">{int(round(pred))} <span class="result-unit">kg/cm²</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No se encontró el archivo 'modelo_M35_1dia_produccion.pkl'.")

# Pie de página institucional
st.markdown("""
<div class="footer-disclaimer">
    <b>QuAI presenta M.A.R.C.O. v1.0</b> (<i>Machine learning Algorithm for Resistance and Cement Optimization</i>) &nbsp;|&nbsp; by Marco Martínez<br>
    QuAI es una IA y puede cometer errores. Las estimaciones son herramientas de soporte operativo y no sustituyen los ensayos físicos oficiales bajo norma ASTM C109.
</div>
""", unsafe_allow_html=True)
