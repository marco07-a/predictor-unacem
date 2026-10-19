import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys
import re
import io
import time
import base64
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================================================================
# 0. CLASES PERSONALIZADAS Y COMPATIBILIDAD SKLEARN / STREAMLIT CLOUD
# ==============================================================================
class IngenieriaQuimicaCemento(BaseEstimator, TransformerMixin):
    def __init__(self, tipo_cemento='HS'):
        self.tipo_cemento = tipo_cemento
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        eps = 0.1
        adiciones = df.get('%Caliza', 0.0) + df.get('%Yeso', 0.0)
        if '%Puzolana' in df.columns:
            adiciones = adiciones + df['%Puzolana']
        if '%Escoria' in df.columns:
            adiciones = adiciones + df['%Escoria']
        df['FCC_Est'] = 100.0 - adiciones
        if 'Alita' in df.columns and 'Belita' in df.columns:
            df['Ratio_Alita_Belita'] = df['Alita'] / (np.abs(df['Belita']) + eps)
        if all(c in df.columns for c in ['SO3', 'C3A_Cúbico', 'C3A_Ortorómbico', 'Álcali_Eq']):
            c3a_tot = np.abs(df['C3A_Cúbico']) + np.abs(df['C3A_Ortorómbico'])
            df['Balance_SO3_C3A'] = df['SO3'] / (c3a_tot + np.abs(df['Álcali_Eq']) + eps)
        if 'Gypsum' in df.columns and 'Hemidrato' in df.columns:
            df['Sulfatos_Solubles'] = df['Gypsum'] + 1.5 * df['Hemidrato']
        if 'Agua_Cemento' in df.columns and 'Blaine' in df.columns:
            df['Indice_Finura_AC'] = df['Blaine'] / (np.abs(df['Agua_Cemento']) + eps)
            df['Potencial_Alita_AC'] = (df['Alita'] * df['Blaine']) / (np.abs(df['Agua_Cemento']) + eps)
        elif 'Blaine' in df.columns and 'Alita' in df.columns:
            df['Potencial_Alita_Blaine'] = (df['Alita'] * df['Blaine']) / 1000.0
        if all(c in df.columns for c in ['%Puzolana', '%Escoria', 'Álcali_Eq']):
            df['Sinergia_SCM'] = (df['%Puzolana'] + df['%Escoria']) * df['Álcali_Eq']
        return df

class CalibradorVarianzaConformal(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, gamma_stretch=0.70, alpha_conformal=0.20, cv=5):
        self.base_estimator = base_estimator
        self.gamma_stretch = gamma_stretch
        self.alpha_conformal = alpha_conformal
        self.cv = cv
    def predict(self, X):
        X_df = pd.DataFrame(X)
        raw_pred = self.model_.predict(X_df)
        return self.mu_y_ + self.factor_escala_ * (raw_pred - self.mu_pred_)
    def predict_interval(self, X):
        y_hat = self.predict(X)
        return y_hat, y_hat - self.q_conformal_, y_hat + self.q_conformal_

setattr(sys.modules["__main__"], "IngenieriaQuimicaCemento", IngenieriaQuimicaCemento)
setattr(sys.modules["__main__"], "CalibradorVarianzaConformal", CalibradorVarianzaConformal)

def reparar_compatibilidad_sklearn(obj):
    if obj is None:
        return None
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = getattr(obj, "_fit_dtype", np.dtype("float64"))
        if not hasattr(obj, "_fit_dtype"):
            obj._fit_dtype = getattr(obj, "_fill_dtype", np.dtype("float64"))
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            reparar_compatibilidad_sklearn(step)
    if hasattr(obj, "estimators_"):
        for est in obj.estimators_:
            reparar_compatibilidad_sklearn(est)
    if hasattr(obj, "model_"):
        reparar_compatibilidad_sklearn(obj.model_)
    if hasattr(obj, "base_estimator"):
        reparar_compatibilidad_sklearn(obj.base_estimator)
    return obj


# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y CARGA INTELIGENTE DE IMÁGENES (PNG / JPEG)
# ==============================================================================
st.set_page_config(
    page_title="DCCAI | UNACEM - División de Control de Calidad Atocongo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def obtener_data_uri_imagen(nombre_base):
    """Detecta automáticamente .png, .jpg o .jpeg y verifica la firma binaria real."""
    for ext in ["", ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        ruta = f"{nombre_base}{ext}"
        if os.path.exists(ruta) and os.path.isfile(ruta):
            with open(ruta, "rb") as f:
                raw = f.read()
            mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
            b64 = base64.b64encode(raw).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    return None

LOGO_URI = obtener_data_uri_imagen("unacem_logo")
BOLSAS_URI = {
    "TI": obtener_data_uri_imagen("sol"),
    "GU": obtener_data_uri_imagen("apu"),
    "HS": obtener_data_uri_imagen("andino")
}

st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 97% !important;
    }
    .splash-box {
        background: radial-gradient(circle at center, #1c222d 0%, #0b0d12 100%);
        border: 1px solid #2e3543;
        border-top: 4px solid #E30613;
        border-radius: 18px;
        padding: 55px 35px;
        text-align: center;
        margin: 45px auto;
        max-width: 820px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.75);
    }
    .splash-dcca {
        font-size: 1.04rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 2.4px;
        text-transform: uppercase;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    .splash-title {
        font-size: 3.3rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }
    .header-banner {
        background: linear-gradient(90deg, #11141b 0%, #191e28 100%);
        border: 1px solid #2d3340;
        border-left: 5px solid #E30613;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
    }
    .brand-tag {
        font-size: 0.78rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .main-title {
        font-size: 1.72rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: 0.5px;
        line-height: 1.15;
    }
    .main-subtitle {
        font-size: 0.84rem;
        color: #9ca3af;
        margin-top: 3px;
    }
    .author-badge {
        background: rgba(227, 6, 19, 0.10);
        border: 1px solid rgba(227, 6, 19, 0.40);
        border-radius: 10px;
        padding: 10px 18px;
        text-align: right;
        font-weight: 700;
        color: #ffffff;
        font-size: 0.92rem;
    }
    .bag-showcase {
        background: #141821;
        border: 1px solid #2d3340;
        border-radius: 10px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        gap: 14px;
        height: 100%;
    }
    .predictor-main-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
        margin-bottom: 4px;
    }
    .section-header {
        font-size: 0.98rem;
        font-weight: 700;
        color: #e5e7eb;
        border-bottom: 1px solid #2d3340;
        padding-bottom: 6px;
        margin-top: 14px;
        margin-bottom: 12px;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #d1d5db !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #E30613 0%, #B8000C 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.04rem;
        padding: 12px 20px;
        box-shadow: 0 4px 15px rgba(227, 6, 19, 0.30);
    }
    .result-card {
        background: radial-gradient(circle at top, #1f242e 0%, #11141a 100%);
        border: 1px solid #303642;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .result-value {
        font-size: 3.4rem;
        font-weight: 800;
        color: #f9fafb;
        line-height: 1.05;
        margin: 4px 0;
    }
    .result-unit {
        font-size: 1.25rem;
        color: #E30613;
        font-weight: 700;
    }
    .conformal-badge {
        display: inline-block;
        margin-top: 12px;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(227, 6, 19, 0.14);
        border: 1px solid rgba(227, 6, 19, 0.4);
        color: #f3f4f6;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .real-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 5px 14px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #ecfdf5;
        font-size: 0.88rem;
        font-weight: 700;
    }
    .kpi-mini {
        background: #151922;
        border-left: 3px solid #E30613;
        border-radius: 6px;
        padding: 9px 14px;
        margin-bottom: 8px;
    }
    .kpi-mini-title {
        font-size: 0.72rem;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .kpi-mini-val {
        font-size: 1.18rem;
        font-weight: 700;
        color: #ffffff;
    }
    .footer-disclaimer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.79rem;
        margin-top: 2.2rem;
        padding-top: 1rem;
        border-top: 1px solid #1f242e;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# PANTALLA DE PRESENTACIÓN INICIAL (SPLASH SCREEN DE 4 SEGUNDOS)
if "splash_mostrado" not in st.session_state:
    placeholder = st.empty()
    logo_html = f'<div><img src="{LOGO_URI}" style="max-height:82px; margin-bottom:10px;"></div>' if LOGO_URI else ''
    bolsas_splash = "".join([
        f'<img src="{uri}" style="height:78px; margin:0 12px; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.5);">'
        for uri in BOLSAS_URI.values() if uri
    ])
    div_bolsas_splash = f'<div style="margin: 18px 0;">{bolsas_splash}</div>' if bolsas_splash else ''

    splash_html = (
        f'<div class="splash-box">'
        f'{logo_html}'
        f'<div class="splash-dcca">DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>'
        f'<div class="splash-title">DCC<span style="color:#E30613;">AI</span></div>'
        f'<div style="color:#d1d5db; font-size:1.1rem; font-weight:500; margin-bottom:14px;">'
        f'Sistema Inteligente de Predicción de Resistencia a la Compresión'
        f'</div>'
        f'{div_bolsas_splash}'
        f'<div style="border-top:1px solid #2d3340; padding-top:16px; font-size:1.0rem; color:#f3f4f6;">'
        f'<b>Elaborado por:</b> Ing. CIP Marco Antonio Uriel Martínez Sánchez'
        f'</div>'
        f'</div>'
    )

    with placeholder.container():
        st.markdown(splash_html, unsafe_allow_html=True)
        barra = st.progress(0, text="Inicializando arquitectura DCCAI...")
        for pct in range(0, 101, 20):
            time.sleep(0.80)
            barra.progress(pct, text="Cargando modelos de Machine Learning (Sol • Apu • Andino)...")
    placeholder.empty()
    st.session_state["splash_mostrado"] = True


# ==============================================================================
# 2. CATÁLOGO DE CEMENTOS Y DICCIONARIO EXACTO DE BASE DE DATOS DE LABORATORIO
# ==============================================================================
CONFIG_CEMENTOS = {
    "TI (Portland Tipo I) - Sol": {
        "codigo": "TI",
        "marca": "Cemento Sol",
        "norma": "ASTM C150 / NTP 334.009",
        "features": [
            '%Caliza', '%Yeso', 'M325', 'Blaine', 'SO3', 'Álcali_Eq',
            'Alita', 'Belita', 'Cal_Libre', 'C3A_Cúbico', 'C3A_Ortorómbico',
            'Gypsum', 'Hemidrato', 'Anhidrita', 'Fluidez'
        ],
        "defaults": {
            '%Caliza': 4.50, '%Yeso': 4.20, '%Puzolana': 0.0, '%Escoria': 0.0,
            'M325': 4.10, 'Blaine': 3650.0, 'SO3': 2.75, 'Álcali_Eq': 0.82,
            'Alita': 56.50, 'Belita': 11.20, 'Cal_Libre': 0.65, 'C3A_Cúbico': 5.80,
            'C3A_Ortorómbico': 1.10, 'Gypsum': 2.50, 'Hemidrato': 1.10,
            'Anhidrita': 0.10, 'Agua_Cemento': 48.50, 'Fluidez': 112.00
        },
        "incertidumbre_base": {"1d": 7.0, "3d": 10.5, "7d": 12.0, "28d": 15.0}
    },
    "GU (General Use) - Apu": {
        "codigo": "GU",
        "marca": "Cemento Apu",
        "norma": "ASTM C1157 / NTP 334.082",
        "features": [
            '%Caliza', '%Yeso', 'M325', 'Blaine', 'SO3', 'Álcali_Eq',
            'Alita', 'Belita', 'Cal_Libre', 'C3A_Cúbico', 'C3A_Ortorómbico',
            'Gypsum', 'Hemidrato', 'Anhidrita', 'Agua_Cemento', 'Fluidez'
        ],
        "defaults": {
            '%Caliza': 21.50, '%Yeso': 4.74, '%Puzolana': 0.0, '%Escoria': 0.0,
            'M325': 2.95, 'Blaine': 3760.0, 'SO3': 2.86, 'Álcali_Eq': 0.77,
            'Alita': 46.20, 'Belita': 7.00, 'Cal_Libre': 0.24, 'C3A_Cúbico': 6.24,
            'C3A_Ortorómbico': 0.00, 'Gypsum': 2.92, 'Hemidrato': 0.78,
            'Anhidrita': 0.05, 'Agua_Cemento': 49.00, 'Fluidez': 110.00
        },
        "incertidumbre_base": {"1d": 6.5, "3d": 9.0, "7d": 11.0, "28d": 14.0}
    },
    "HS (High Sulfate Resistance) - Andino": {
        "codigo": "HS",
        "marca": "Cemento Andino Ultra HS",
        "norma": "ASTM C1157 / NTP 334.082",
        "features": [
            '%Caliza', '%Yeso', '%Puzolana', '%Escoria', 'M325', 'Blaine',
            'SO3', 'Álcali_Eq', 'Alita', 'Belita', 'Cal_Libre', 'C3A_Cúbico',
            'C3A_Ortorómbico', 'Gypsum', 'Hemidrato', 'Anhidrita',
            'Agua_Cemento', 'Fluidez'
        ],
        "defaults": {
            '%Caliza': 10.00, '%Yeso': 4.00, '%Puzolana': 15.00, '%Escoria': 18.00,
            'M325': 0.18, 'Blaine': 4960.0, 'SO3': 2.50, 'Álcali_Eq': 1.37,
            'Alita': 44.50, 'Belita': 9.30, 'Cal_Libre': 0.10, 'C3A_Cúbico': 2.40,
            'C3A_Ortorómbico': 0.00, 'Gypsum': 2.30, 'Hemidrato': 2.10,
            'Anhidrita': 0.20, 'Agua_Cemento': 49.00, 'Fluidez': 110.00
        },
        "incertidumbre_base": {"1d": 6.0, "3d": 9.8, "7d": 11.5, "28d": 14.1}
    }
}

MAPEO_BD_EXACTO = {
    'cal': '%Caliza', 'caliza': '%Caliza', 'porcentajecaliza': '%Caliza',
    'yes': '%Yeso', 'yeso': '%Yeso', 'porcentajeyeso': '%Yeso',
    'puzare': '%Puzolana', 'puz': '%Puzolana', 'puzolana': '%Puzolana',
    'slag': '%Escoria', 'escoria': '%Escoria',
    'm325prvs': 'M325', 'm325': 'M325',
    'blaineaut': 'Blaine', 'blaine': 'Blaine',
    'so3rxpa': 'SO3', 'so3': 'SO3',
    'alctotrxpa': 'Álcali_Eq', 'alctot': 'Álcali_Eq', 'alcalieq': 'Álcali_Eq',
    'alita': 'Alita', 'c3s': 'Alita',
    'belita': 'Belita', 'c2s': 'Belita',
    'callibredrx': 'Cal_Libre', 'callibre': 'Cal_Libre', 'cao': 'Cal_Libre',
    'c3acubic': 'C3A_Cúbico', 'c3acubico': 'C3A_Cúbico',
    'c3aortho': 'C3A_Ortorómbico', 'c3aortorombico': 'C3A_Ortorómbico', 'c3aortorrombico': 'C3A_Ortorómbico',
    'gypsum': 'Gypsum',
    'hemidrato': 'Hemidrato', 'hemihidrato': 'Hemidrato',
    'anhydrita': 'Anhidrita', 'anhidrita': 'Anhidrita',
    'agcem': 'Agua_Cemento', 'aguacemento': 'Agua_Cemento',
    'fluid': 'Fluidez', 'fluidez': 'Fluidez',
    '24hrs': '1_d', '24h': '1_d', '1d': '1_d', '1dia': '1_d',
    '3dias': '3_d', '3d': '3_d',
    '7d': '7_d', '7dias': '7_d',
    '28dias': '28_d', '28d': '28_d'
}


# ==============================================================================
# 3. CARGA DE MODELOS (.PKL)
# ==============================================================================
@st.cache_resource
def cargar_repositorio_modelos():
    candidatos = {
        "GU": {
            "1d": ["modelo_GU_1d_produccion.pkl", "modelo_M35_1dia_produccion.pkl"],
            "3d": ["modelo_GU_3d_produccion.pkl", "modelo_M35_3dias_produccion.pkl"],
            "7d": ["modelo_GU_7d_produccion.pkl", "modelo_M35_7dias_produccion.pkl"],
            "28d": ["modelo_GU_28d_PRO_produccion.pkl", "modelo_GU_28d_produccion.pkl", "modelo_M34_28d_produccion.pkl"],
            "28d_CASCADA": ["modelo_GU_28d_CASCADA_produccion.pkl"]
        },
        "TI": {
            "1d": ["modelo_TI_1d_produccion.pkl"],
            "3d": ["modelo_TI_3d_produccion.pkl"],
            "7d": ["modelo_TI_7d_produccion.pkl"],
            "28d": ["modelo_TI_28d_PRO_produccion.pkl", "modelo_TI_28d_produccion.pkl"],
            "28d_CASCADA": ["modelo_TI_28d_CASCADA_produccion.pkl"]
        },
        "HS": {
            "1d": ["modelo_HS_1d_produccion.pkl"],
            "3d": ["modelo_HS_3d_produccion.pkl"],
            "7d": ["modelo_HS_7d_produccion.pkl"],
            "28d": ["modelo_HS_28d_PRO_produccion.pkl", "modelo_HS_28d_produccion.pkl"],
            "28d_CASCADA": ["modelo_HS_28d_CASCADA_produccion.pkl"]
        }
    }
    repo = {"GU": {}, "TI": {}, "HS": {}}
    for cem, edades in candidatos.items():
        for edad, rutas in edades.items():
            modelo_cargado, ruta_activa = None, None
            for r in rutas:
                if os.path.exists(r):
                    try:
                        raw_mod = joblib.load(r)
                        modelo_cargado = reparar_compatibilidad_sklearn(raw_mod)
                        ruta_activa = r
                        break
                    except Exception:
                        pass
            repo[cem][edad] = (modelo_cargado, ruta_activa)
    return repo

REPO_MODELOS = cargar_repositorio_modelos()


# ==============================================================================
# 4. MOTOR ETL: LIMPIEZA EXACTA DEL EXCEL EXPORTADO DE BASE DE DATOS
# ==============================================================================
def normalizar_nombre(texto):
    return re.sub(
        r'[^a-z0-9]', '',
        unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()
    )

def parsear_fecha_robusta(serie_fecha):
    """Convierte fechas de Excel incluso cuando vienen con '12:00 a. m.' o 'p. m.' en texto."""
    limpia = (
        serie_fecha.astype(str)
        .str.replace(r'\s+12:00\s*a\.?\s*m\.?', '', flags=re.IGNORECASE, regex=True)
        .str.replace(r'a\.?\s*m\.?', 'AM', flags=re.IGNORECASE, regex=True)
        .str.replace(r'p\.?\s*m\.?', 'PM', flags=re.IGNORECASE, regex=True)
        .str.strip()
    )
    return pd.to_datetime(limpia, errors='coerce')

def limpiar_dataframe_lotes(df_raw, features_esperadas, defaults_cem, edad_seleccionada=None):
    """
    1. Trunca desde la fila 'Promedio / Desviación Estandar' hacia abajo.
    2. Elimina todas las columnas a la izquierda de 'Fecha' (N°, ALSE/FALSE y blancos).
    3. Renombra los encabezados crudos (%Cal, % PuzAre, % Slag, 24 hrs...) al estándar del modelo.
    4. Conserva únicamente la columna de la edad seleccionada (1_d, 3_d, 7_d o 28_d).
    """
    df = df_raw.copy()

    # PASO 1: Cortar desde la fila "Promedio / Desviación Estandar" hacia abajo
    idx_corte = None
    for i in range(len(df)):
        fila_texto = " ".join(df.iloc[i].astype(str).tolist()).lower()
        if any(p in fila_texto for p in ["promedio", "desviacion", "desviación", "estandar", "estándar"]):
            idx_corte = i
            break
    if idx_corte is not None:
        df = df.iloc[:idx_corte].copy()

    # PASO 2: Localizar 'Fecha' y descartar las columnas previas (N°, ALSE/FALSE, blancos)
    idx_col_fecha = None
    for j, col_name in enumerate(df.columns):
        if normalizar_nombre(col_name) in ['fecha', 'date', 'fechamolienda']:
            idx_col_fecha = j
            break
    if idx_col_fecha is not None:
        df = df.iloc[:, idx_col_fecha:].copy()

    # PASO 3: Renombrar encabezados crudos al estándar exacto del modelo
    renombres = {}
    for col in df.columns:
        c_norm = normalizar_nombre(col)
        if c_norm in MAPEO_BD_EXACTO:
            renombres[col] = MAPEO_BD_EXACTO[c_norm]
        elif 'fecha' in c_norm:
            renombres[col] = 'Fecha'
    df = df.rename(columns=renombres)

    # PASO 4: Filtrar cualquier fila sin Fecha válida (o filas vacías si ya venía recortado)
    if 'Fecha' in df.columns:
        fechas_dt = parsear_fecha_robusta(df['Fecha'])
        df = df[fechas_dt.notna()].copy()
        df['Fecha'] = fechas_dt[fechas_dt.notna()].dt.strftime('%Y-%m-%d')
    else:
        cols_num = [c for c in features_esperadas if c in df.columns]
        if cols_num:
            df = df.dropna(subset=cols_num, how='all').copy()

    # PASO 5: Construir DataFrame numérico para el modelo
    df_modelo = pd.DataFrame(index=df.index)
    cols_receta_fija = ['%Caliza', '%Yeso', '%Puzolana', '%Escoria']
    for col in features_esperadas:
        if col in df.columns:
            serie_num = pd.to_numeric(df[col], errors='coerce')
            if col in cols_receta_fija:
                serie_num = serie_num.ffill().fillna(defaults_cem.get(col, 0.0))
            df_modelo[col] = serie_num
            df[col] = serie_num
        else:
            df_modelo[col] = defaults_cem.get(col, 0.0)

    # PASO 6: Conservar solo Fecha, variables predictoras y la edad activa
    columnas_vista = []
    if 'Fecha' in df.columns:
        columnas_vista.append('Fecha')
    columnas_vista.extend([c for c in features_esperadas if c in df.columns])

    if edad_seleccionada in ['1d', '3d', '7d', '28d']:
        target_activo = f"{edad_seleccionada[:-1]}_d"
        if target_activo in df.columns:
            df[target_activo] = pd.to_numeric(df[target_activo], errors='coerce')
            columnas_vista.append(target_activo)

    df_vista_final = df[columnas_vista].reset_index(drop=True)
    return df_vista_final, df_modelo.reset_index(drop=True)

def ejecutar_predict_seguro(modelo, df_aln):
    reparar_compatibilidad_sklearn(modelo)
    try:
        return np.asarray(modelo.predict(df_aln), dtype=float)
    except AttributeError:
        X_val = df_aln.select_dtypes(include=[np.number]).fillna(-9999.0).values
        if hasattr(modelo, "steps"):
            for _, paso in modelo.steps[:-1]:
                if not isinstance(paso, SimpleImputer):
                    X_val = paso.transform(X_val)
            return np.asarray(modelo.steps[-1][1].predict(X_val), dtype=float)
        raise

def predecir_con_modelo(modelo, df_features, features_esperadas, sigma_fallback=10.0):
    df_aln = df_features[features_esperadas].copy()
    if hasattr(modelo, "predict_interval"):
        try:
            reparar_compatibilidad_sklearn(modelo)
            y_hat, p10, p90 = modelo.predict_interval(df_aln)
            return np.asarray(y_hat, dtype=float), np.asarray(p10, dtype=float), np.asarray(p90, dtype=float)
        except AttributeError:
            raw_pred = ejecutar_predict_seguro(modelo.model_, df_aln)
            y_hat = modelo.mu_y_ + modelo.factor_escala_ * (raw_pred - modelo.mu_pred_)
            q = getattr(modelo, "q_conformal_", 1.2816 * sigma_fallback)
            return y_hat, y_hat - q, y_hat + q
    else:
        y_hat = ejecutar_predict_seguro(modelo, df_aln)
        q_80 = 1.2816 * sigma_fallback
        return y_hat, y_hat - q_80, y_hat + q_80

def obtener_importancia_variables(modelo, df_fila, features_esperadas):
    est = modelo.model_ if hasattr(modelo, "model_") else modelo
    if hasattr(est, "steps"):
        est = est.steps[-1][1]

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        if len(imp) == len(features_esperadas) and imp.max() > 0:
            imp_pct = (imp / imp.sum()) * 100.0
            return pd.DataFrame({'Variable': features_esperadas, 'Importancia (%)': imp_pct})

    pred_0, _, _ = predecir_con_modelo(modelo, df_fila, features_esperadas)
    sens = []
    for col in features_esperadas:
        df_p = df_fila.copy()
        delta = max(abs(float(df_p[col].values[0])) * 0.05, 0.2)
        df_p[col] = df_p[col] + delta
        pred_1, _, _ = predecir_con_modelo(modelo, df_p, features_esperadas)
        sens.append(abs(float(pred_1[0] - pred_0[0])) + 0.05)
    sens = np.asarray(sens)
    imp_pct = (sens / sens.sum()) * 100.0
    return pd.DataFrame({'Variable': features_esperadas, 'Importancia (%)': imp_pct})


# ==============================================================================
# 5. ENCABEZADO INSTITUCIONAL Y SELECTORES CON IMAGEN DE BOLSA DE CEMENTO
# ==============================================================================
logo_header_html = f'<img src="{LOGO_URI}" style="height:50px; margin-right:18px;">' if LOGO_URI else ''

header_html = (
    f'<div class="header-banner">'
    f'<div style="display:flex; align-items:center;">'
    f'{logo_header_html}'
    f'<div>'
    f'<div class="brand-tag">DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>'
    f'<div class="main-title">DCC<span style="color:#E30613;">AI</span> &mdash; Inteligencia Artificial Aplicada al Cemento</div>'
    f'<div class="main-subtitle">Sistema Inteligente de Predicción de Resistencia a la Compresión</div>'
    f'</div>'
    f'</div>'
    f'<div class="author-badge">Ing. CIP Marco Antonio Uriel Martínez Sánchez</div>'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

col_sel1, col_sel2, col_sel3, col_bag = st.columns([1.35, 1.25, 1.0, 1.1])
with col_sel1:
    tipo_cemento_label = st.selectbox("Cemento", list(CONFIG_CEMENTOS.keys()), index=1)
with col_sel2:
    edad_label = st.selectbox(
        "Edad de Predicción",
        [
            "1 día",
            "3 días",
            "7 días",
            "28 días",
            "Vista multi-edad completa (1, 3, 7 y 28 días)"
        ],
        index=3
    )
with col_sel3:
    cfg = CONFIG_CEMENTOS[tipo_cemento_label]
    cod_cem = cfg["codigo"]
    st.write("")
    usar_cascada_28d = st.toggle("Modo Cascada 28d", value=False)

with col_bag:
    bag_uri = BOLSAS_URI.get(cod_cem)
    img_bag_top = f'<img src="{bag_uri}" style="height:58px; border-radius:4px;">' if bag_uri else ''
    st.markdown(
        f'<div class="bag-showcase">{img_bag_top}'
        f'<div><div style="font-size:0.72rem; color:#E30613; font-weight:800; text-transform:uppercase;">{cfg["norma"]}</div>'
        f'<div style="font-size:0.95rem; color:#ffffff; font-weight:800;">{cfg["marca"]}</div></div></div>',
        unsafe_allow_html=True
    )


# ==============================================================================
# 6. PESTAÑAS EJECUTIVAS DE MACHINE LEARNING
# ==============================================================================
tab_sim, tab_batch, tab_spc = st.tabs([
    "🎛️ Inferencia & Variables Predictoras",
    "📂 Carga por Lotes & Métricas",
    "📈 Diagnóstico de Residuales & Control Estadístico"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: INFERENCIA & VARIABLES PREDICTORAS
# ------------------------------------------------------------------------------
with tab_sim:
    defaults = cfg["defaults"].copy()
    features_activas = cfg["features"]
    valor_real_lote = None

    if f"lotes_{cod_cem}" in st.session_state:
        df_mem_vista, df_mem_mod = st.session_state[f"lotes_{cod_cem}"]
        if len(df_mem_vista) > 0:
            opciones_lote = ["Ingreso manual libre"] + [
                f"Lote {i+1} | Fecha: {df_mem_vista['Fecha'].iloc[i]}" if 'Fecha' in df_mem_vista.columns else f"Lote #{i+1}"
                for i in range(len(df_mem_vista))
            ]
            lote_elegido = st.selectbox("🔍 Sincronizar variables y Valor Real desde un lote del Excel cargado:", opciones_lote)
            if lote_elegido != "Ingreso manual libre":
                idx_lote = opciones_lote.index(lote_elegido) - 1
                for f_col in features_activas:
                    if f_col in df_mem_mod.columns and pd.notna(df_mem_mod[f_col].iloc[idx_lote]):
                        defaults[f_col] = float(df_mem_mod[f_col].iloc[idx_lote])
                mapa_ed_col = {"1 día": "1_d", "3 días": "3_d", "7 días": "7_d", "28 días": "28_d"}
                col_real_buscada = mapa_ed_col.get(edad_label)
                if col_real_buscada and col_real_buscada in df_mem_vista.columns:
                    val_r = pd.to_numeric(df_mem_vista[col_real_buscada].iloc[idx_lote], errors='coerce')
                    if pd.notna(val_r):
                        valor_real_lote = float(val_r)

    st.markdown(
        f'<div class="predictor-main-title">Variables Predictoras &mdash; <span style="color:#E30613;">{cfg["marca"]}</span></div>',
        unsafe_allow_html=True
    )

    # 1. DIFRACCIÓN DE RAYOS X (MÉTODO RIETVELD)
    st.markdown('<div class="section-header">🔬 1. Difracción de Rayos X (Método Rietveld)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alita = st.number_input("Alita (C3S)", value=float(defaults['Alita']), step=0.50, format="%.2f")
        c3a_cub = st.number_input("C3A Cúbico", value=float(defaults['C3A_Cúbico']), step=0.10, format="%.2f")
    with c2:
        belita = st.number_input("Belita (C2S)", value=float(defaults['Belita']), step=0.50, format="%.2f")
        c3a_orto = st.number_input("C3A Ortorrómbico", value=float(defaults['C3A_Ortorómbico']), step=0.10, format="%.2f")
    with c3:
        gypsum = st.number_input("Gypsum", value=float(defaults['Gypsum']), step=0.10, format="%.2f")
        cal_libre = st.number_input("Cal Libre", value=float(defaults['Cal_Libre']), step=0.05, format="%.2f")
    with c4:
        hemidrato = st.number_input("Hemidrato", value=float(defaults['Hemidrato']), step=0.10, format="%.2f")
        anhidrita = st.number_input("Anhidrita", value=float(defaults['Anhidrita']), step=0.05, format="%.2f")

    # 2. DOSIFICACIÓN Y FLUORESCENCIA DE RAYOS X
    st.markdown('<div class="section-header">⚗️ 2. Dosificación y Fluorescencia de Rayos X</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        caliza = st.number_input("%Caliza", value=float(defaults['%Caliza']), step=0.10, format="%.2f")
    with f2:
        yeso = st.number_input("%Yeso", value=float(defaults['%Yeso']), step=0.10, format="%.2f")
    with f3:
        so3 = st.number_input("SO3 (%)", value=float(defaults['SO3']), step=0.05, format="%.2f")
    with f4:
        alcali_eq = st.number_input("Álcali_Eq", value=float(defaults['Álcali_Eq']), step=0.01, format="%.2f")

    if cod_cem == "HS":
        h1, h2, _, _ = st.columns(4)
        with h1:
            puzolana = st.number_input("%Puzolana", value=float(defaults['%Puzolana']), step=0.20, format="%.2f")
        with h2:
            escoria = st.number_input("%Escoria", value=float(defaults['%Escoria']), step=0.20, format="%.2f")
    else:
        puzolana, escoria = 0.0, 0.0

    # 3. CONTROL DE PROCESOS Y FÍSICOS
    st.markdown('<div class="section-header">⚙️ 3. Control de Procesos y Físicos</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        m325 = st.number_input("M325 (%)", value=float(defaults['M325']), step=0.05, format="%.2f")
    with p2:
        blaine = st.number_input("Blaine (cm²/g)", value=float(defaults['Blaine']), step=10.0, format="%.1f")
    with p3:
        if cod_cem in ["GU", "HS"]:
            agua_cemento = st.number_input("Agua/Cemento (%)", value=float(defaults['Agua_Cemento']), step=0.20, format="%.2f")
        else:
            agua_cemento = 48.50
            st.caption("ℹ️ *Relación A/C estándar en Tipo I: 48.5%*")
    with p4:
        fluidez = st.number_input("Fluidez (%)", value=float(defaults['Fluidez']), step=1.0, format="%.1f")

    res_1d_real, res_3d_real = None, None
    if usar_cascada_28d:
        cc1, cc2 = st.columns(2)
        with cc1:
            res_1d_real = st.number_input("Resistencia Real 1 día (kg/cm²)", value=140.0, step=1.0)
        with cc2:
            res_3d_real = st.number_input("Resistencia Real 3 días (kg/cm²)", value=252.0, step=1.0)

    registro_actual = {
        '%Caliza': caliza, '%Yeso': yeso, '%Puzolana': puzolana, '%Escoria': escoria,
        'M325': m325, 'Blaine': blaine, 'SO3': so3, 'Álcali_Eq': alcali_eq,
        'Alita': alita, 'Belita': belita, 'Cal_Libre': cal_libre,
        'C3A_Cúbico': c3a_cub, 'C3A_Ortorómbico': c3a_orto,
        'Gypsum': gypsum, 'Hemidrato': hemidrato, 'Anhidrita': anhidrita,
        'Agua_Cemento': agua_cemento, 'Fluidez': fluidez
    }
    if usar_cascada_28d:
        registro_actual['1_d'] = res_1d_real
        registro_actual['3_d'] = res_3d_real

    df_input_single = pd.DataFrame([registro_actual])

    # 4. INDICADORES FISICOQUÍMICOS DERIVADOS
    st.markdown('<div class="section-header">📊 4. Indicadores Fisicoquímicos Derivados</div>', unsafe_allow_html=True)
    fcc_real = 100.0 - (caliza + yeso + puzolana + escoria)
    ratio_silicatos = alita / (belita + 0.1)
    sulf_solubles = gypsum + 1.5 * hemidrato
    c3a_total = c3a_cub + c3a_orto
    balance_so3 = so3 / (c3a_total + alcali_eq + 0.1)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-mini"><div class="kpi-mini-title">Factor Clinker (FCC)</div><div class="kpi-mini-val">{fcc_real:.2f}%</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-mini"><div class="kpi-mini-title">Ratio Alita / Belita</div><div class="kpi-mini-val">{ratio_silicatos:.2f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-mini"><div class="kpi-mini-title">Sulfatos Solubles</div><div class="kpi-mini-val">{sulf_solubles:.2f}%</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-mini"><div class="kpi-mini-title">Balance SO3 / (C3A+Álc)</div><div class="kpi-mini-val">{balance_so3:.2f}</div></div>', unsafe_allow_html=True)

    if st.button("Ejecutar Predicción", use_container_width=True):
        mapa_edad = {"1 día": "1d", "3 días": "3d", "7 días": "7d", "28 días": "28d"}
        edad_elegida = mapa_edad.get(edad_label, "MULTI")

        if edad_elegida != "MULTI":
            clave_m = "28d_CASCADA" if (edad_elegida == "28d" and usar_cascada_28d and REPO_MODELOS[cod_cem]["28d_CASCADA"][0] is not None) else edad_elegida
            mod_obj, _ = REPO_MODELOS[cod_cem][clave_m]
            sigma_fb = cfg["incertidumbre_base"][edad_elegida]

            if mod_obj is not None:
                cols_req = features_activas + (['1_d', '3_d'] if clave_m == "28d_CASCADA" else [])
                y_p, p10, p90 = predecir_con_modelo(mod_obj, df_input_single, cols_req, sigma_fallback=sigma_fb)
                val_c, val_p10, val_p90 = float(y_p[0]), float(p10[0]), float(p90[0])

                html_real = ""
                if valor_real_lote is not None:
                    desvio = abs(valor_real_lote - val_c)
                    html_real = f'<div><span class="real-badge">✔️ Valor Real: {valor_real_lote:.0f} kg/cm² (MAE: {desvio:.1f} kg/cm²)</span></div>'

                bag_card_html = f'<div style="margin-bottom:8px;"><img src="{bag_uri}" style="height:64px; border-radius:4px;"></div>' if bag_uri else ''

                card_html = (
                    f'<div class="result-card">'
                    f'{bag_card_html}'
                    f'<div style="font-size:0.82rem; color:#9ca3af; text-transform:uppercase; margin-bottom:6px;">'
                    f'{tipo_cemento_label} &bull; {edad_label}'
                    f'</div>'
                    f'<div class="result-value">{int(round(val_c))} <span class="result-unit">kg/cm²</span></div>'
                    f'{html_real}'
                    f'<div><span class="conformal-badge">'
                    f'🛡️ Intervalo Conformal (P10 – P90): <b>{val_p10:.0f}</b> a <b>{val_p90:.0f} kg/cm²</b>'
                    f'</span></div>'
                    f'</div>'
                )

                col_res, col_imp = st.columns([1.0, 1.4])
                with col_res:
                    st.markdown(card_html, unsafe_allow_html=True)

                with col_imp:
                    df_imp = obtener_importancia_variables(mod_obj, df_input_single, features_activas)
                    df_imp = df_imp.sort_values('Importancia (%)', ascending=True).tail(8)

                    fig_imp = go.Figure(go.Bar(
                        x=df_imp['Importancia (%)'], y=df_imp['Variable'],
                        orientation='h', marker_color='#E30613',
                        text=[f"{v:.1f}%" for v in df_imp['Importancia (%)']],
                        textposition='auto'
                    ))
                    fig_imp.update_layout(
                        title="Importancia de Variables Predictoras (Feature Importance)",
                        template="plotly_dark", height=290,
                        margin=dict(l=10, r=10, t=35, b=10),
                        xaxis_title="Peso Relativo en el Modelo (%)"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.error(f"No se encontró el modelo `.pkl` para **{cod_cem} - {edad_elegida}**.")
        else:
            edades_seq = ["1d", "3d", "7d", "28d"]
            etiquetas_ed = ["1 día", "3 días", "7 días", "28 días"]
            dias_num = [1, 3, 7, 28]
            preds_c, preds_p10, preds_p90 = [], [], []
            cols_cards = st.columns(4)

            for idx, ed in enumerate(edades_seq):
                mod_e, _ = REPO_MODELOS[cod_cem][ed]
                sigma_e = cfg["incertidumbre_base"][ed]
                if mod_e is not None:
                    yp, l10, l90 = predecir_con_modelo(mod_e, df_input_single, features_activas, sigma_fallback=sigma_e)
                    preds_c.append(float(yp[0]))
                    preds_p10.append(float(l10[0]))
                    preds_p90.append(float(l90[0]))
                    mini_card_html = (
                        f'<div class="result-card" style="padding:14px;">'
                        f'<div style="font-size:0.78rem; color:#9ca3af;">{etiquetas_ed[idx].upper()}</div>'
                        f'<div class="result-value" style="font-size:2.3rem;">{int(round(yp[0]))} <span class="result-unit" style="font-size:0.95rem;">kg/cm²</span></div>'
                        f'<div style="font-size:0.74rem; color:#9ca3af; margin-top:4px;">P10–P90: [{l10[0]:.0f} – {l90[0]:.0f}]</div>'
                        f'</div>'
                    )
                    cols_cards[idx].markdown(mini_card_html, unsafe_allow_html=True)
                else:
                    preds_c.append(np.nan)
                    preds_p10.append(np.nan)
                    preds_p90.append(np.nan)

            if not all(np.isnan(preds_c)):
                fig_curva = go.Figure()
                fig_curva.add_trace(go.Scatter(
                    x=dias_num + dias_num[::-1],
                    y=preds_p90 + preds_p10[::-1],
                    fill='toself', fillcolor='rgba(227, 6, 19, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalo Conformal P10–P90'
                ))
                fig_curva.add_trace(go.Scatter(
                    x=dias_num, y=preds_c, mode='lines+markers+text',
                    line=dict(color='#E30613', width=3),
                    marker=dict(size=9, color='#ffffff', line=dict(width=2, color='#E30613')),
                    text=[f"{v:.0f}" if not np.isnan(v) else "" for v in preds_c],
                    textposition="top center", name='Evolución Predicha'
                ))
                fig_curva.update_layout(
                    title=f"Curva Multi-Edad de Evolución de Resistencia — {tipo_cemento_label}",
                    xaxis_title="Edad (días)", yaxis_title="Resistencia (kg/cm²)",
                    template="plotly_dark", height=330
                )
                st.plotly_chart(fig_curva, use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 2: CARGA POR LOTES & MÉTRICAS (LIMPIEZA AUTOMÁTICA DE BD EN BRUTO)
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown("##### 📂 Carga por Lotes y Resultado de Métricas (`.xlsx`)")

    mapa_horizonte_batch = {
        "1 día": "1d",
        "3 días": "3d",
        "7 días": "7d",
        "28 días": "28d"
    }
    edad_batch = mapa_horizonte_batch.get(edad_label, "28d")
    target_col = f"{edad_batch[:-1]}_d"

    st.caption(
        f"Edad activa: **{edad_batch[:-1]} días**. El motor elimina automáticamente las columnas previas a `Fecha` (`N°`, `ALSE`), "
        f"transforma los encabezados de laboratorio, descarta las demás edades y corta las filas finales de `Promedio / Desviación Estandar`."
    )
    archivo_subido = st.file_uploader(f"Sube el archivo Excel exportado de la base de datos ({tipo_cemento_label})", type=["xlsx"])

    if archivo_subido is not None:
        df_raw = pd.read_excel(archivo_subido)
        df_vista, df_mod = limpiar_dataframe_lotes(
            df_raw, cfg["features"], cfg["defaults"], edad_seleccionada=edad_batch
        )
        st.session_state[f"lotes_{cod_cem}"] = (df_vista, df_mod)

        mod_b, _ = REPO_MODELOS[cod_cem][edad_batch]
        if mod_b is not None and len(df_mod) > 0:
            y_b, p10_b, p90_b = predecir_con_modelo(
                mod_b, df_mod, cfg["features"], sigma_fallback=cfg["incertidumbre_base"][edad_batch]
            )
            df_out = df_vista.copy()
            df_out.insert(0, f"{edad_batch}_PREDICTION", np.round(y_b, 0))

            if target_col in df_out.columns:
                y_real_b = pd.to_numeric(df_out[target_col], errors='coerce')
                mask_b = y_real_b.notna()

                df_out["MAE"] = np.where(mask_b, np.round(np.abs(y_real_b - y_b), 2), np.nan)
                df_out["MAPE (%)"] = np.where(mask_b, np.round((df_out["MAE"] / y_real_b) * 100, 2), np.nan)
                df_out["RMSE"] = np.where(mask_b, np.round((y_real_b - y_b) ** 2, 2), np.nan)

                if mask_b.sum() > 0:
                    mae_g = mean_absolute_error(y_real_b[mask_b], y_b[mask_b])
                    mape_g = np.mean(np.abs((y_real_b[mask_b] - y_b[mask_b]) / y_real_b[mask_b])) * 100
                    rmse_g = np.sqrt(mean_squared_error(y_real_b[mask_b], y_b[mask_b]))
                    st.session_state["ultimo_df_validado"] = (df_out, y_real_b[mask_b], y_b[mask_b], edad_batch)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE Promedio", f"{mae_g:.2f} kg/cm²")
                    m2.metric("MAPE Promedio", f"{mape_g:.2f}%")
                    m3.metric("RMSE Global", f"{rmse_g:.2f} kg/cm²")
                    m4.metric("Lotes Procesados", f"{mask_b.sum()} reales | {(~mask_b).sum()} en curado")

            st.dataframe(df_out, use_container_width=True, height=360)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_out.to_excel(writer, index=False, sheet_name=f"DCCAI_{cod_cem}_{edad_batch}")
            st.download_button(
                label=f"⬇️ Descargar Reporte de Métricas ({cod_cem} - {edad_batch})",
                data=buffer.getvalue(),
                file_name=f"DCCAI_Resultados_{cod_cem}_{edad_batch}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.error(f"No se pudo cargar el archivo `.pkl` de **{cod_cem} - {edad_batch}**.")

# ------------------------------------------------------------------------------
# PESTAÑA 3: DIAGNÓSTICO DE RESIDUALES & CONTROL ESTADÍSTICO
# ------------------------------------------------------------------------------
with tab_spc:
    st.markdown("##### 📈 Diagnóstico de Residuales y Control Estadístico")
    if "ultimo_df_validado" in st.session_state:
        df_v, y_r, y_p, ed_v = st.session_state["ultimo_df_validado"]
        residuales = y_p - y_r
        pendiente = np.polyfit(y_r, residuales, 1)[0]

        g1, g2 = st.columns(2)
        with g1:
            fig_sc = px.scatter(
                x=y_r, y=y_p, labels={'x': 'Valor Real (kg/cm²)', 'y': 'Predicción DCCAI (kg/cm²)'},
                title=f"Predicción vs Valor Real ({cod_cem} - {ed_v})"
            )
            min_v, max_v = min(y_r.min(), y_p.min()), max(y_r.max(), y_p.max())
            fig_sc.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="#E30613", dash="dash"))
            fig_sc.update_layout(template="plotly_dark", height=330)
            st.plotly_chart(fig_sc, use_container_width=True)

        with g2:
            fig_res = px.scatter(
                x=y_r, y=residuales, labels={'x': 'Valor Real (kg/cm²)', 'y': 'Residual (Predicho - Real)'},
                title=f"Distribución de Residuales (Pendiente = {pendiente:.3f})"
            )
            fig_res.add_hline(y=0, line_color="white", line_dash="solid")
            fig_res.update_layout(template="plotly_dark", height=330)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Carga un archivo en la pestaña **Carga por Lotes & Métricas** para visualizar el diagnóstico de residuales.")

# ==============================================================================
# 7. PIE DE PÁGINA INSTITUCIONAL
# ==============================================================================
footer_html = (
    '<div class="footer-disclaimer">'
    '<div><b>UNACEM &mdash; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO (DCCAI)</b> &nbsp;|&nbsp; '
    '<b>Elaborado por:</b> Ing. CIP Marco Antonio Uriel Martínez Sánchez</div>'
    '<div>Las estimaciones e intervalos conformales son herramientas de soporte operativo y pueden presentar variaciones de acuerdo al proceso.</div>'
    '</div>'
)
st.markdown(footer_html, unsafe_allow_html=True)
