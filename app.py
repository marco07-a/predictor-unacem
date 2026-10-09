import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="UNACEM - Predictor GU",
    page_icon="⚡",
    layout="wide"
)

# Estilos de tarjeta y botones
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e222a 0%, #16191f 100%);
        border: 1px solid #2d3340;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-unit {
        font-size: 1.2rem;
        color: #e63946;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 8px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #e63946, #d90429);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 12px 24px;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    m1 = None
    m28 = None
    for f in ["modelo_GU_1dia.pkl", "modelo_xgboost_gu.pkl"]:
        if os.path.exists(f):
            m1 = joblib.load(f)
            break
    if os.path.exists("modelo_M34_28d_produccion.pkl"):
        m28 = joblib.load("modelo_M34_28d_produccion.pkl")
    return m1, m28

mod_1d, mod_28d = cargar_modelos()

# Encabezado
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    if os.path.exists("unacem_logo.png"):
        st.image("unacem_logo.png", width=120)
with header_col2:
    st.title("Sistema Predictivo de Resistencia")
    st.caption("Cemento Tipo GU | Control Fisicoquímico de Producción")

st.write("")

# Selector de horizonte
sel_col1, sel_col2 = st.columns([2, 4])
with sel_col1:
    horizonte = st.selectbox(
        "Horizonte de Predicción",
        ["28 Días (Resistencia Terminal)", "1 Día (Resistencia Temprana)"]
    )
with sel_col2:
    if "28 Días" in horizonte:
        st.caption("Modelo activo: **LightGBM M34** | Error medio: **±9.27 kg/cm²** (MAPE: 2.58%)")
    else:
        st.caption("Modelo activo: **XGBoost** | Error medio: **±5.67 kg/cm²**")

st.markdown("---")

# Entradas (4 columnas limpias)
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
            pred = float(mod_28d.predict(input_data)[0])
            mae = 9.27
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{pred:.1f} <span class="metric-unit">kg/cm²</span></div>
                <div class="metric-sub">Rango estimado (±1 MAE): <b>{pred - mae:.1f} — {pred + mae:.1f} kg/cm²</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No se encontró 'modelo_M34_28d_produccion.pkl' en el repositorio.")
    else:
        if mod_1d is not None:
            pred = float(mod_1d.predict(input_data)[0])
            mae = 5.67
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{pred:.1f} <span class="metric-unit">kg/cm²</span></div>
                <div class="metric-sub">Rango estimado (±1 MAE): <b>{pred - mae:.1f} — {pred + mae:.1f} kg/cm²</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No se encontró el modelo de 1 día en el repositorio.")
