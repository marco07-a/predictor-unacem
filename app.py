import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
# 0. CLASES PERSONALIZADAS Y COMPATIBILIDAD SKLEARN / XGBOOST / STREAMLIT CLOUD
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
# 1. CONFIGURACIÓN DE PÁGINA Y ESCALA TIPOGRÁFICA BALANCEADA
# ==============================================================================
st.set_page_config(
    page_title="DCCAI: Inteligencia Artificial para el Control de Calidad | UNACEM",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def obtener_data_uri_imagen(nombre_base):
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
    .stApp {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%) !important;
    }
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 97% !important;
    }
    .splash-box {
        background: radial-gradient(circle at center, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-top: 4px solid #E30613;
        border-radius: 18px;
        padding: 50px 35px;
        text-align: center;
        margin: 40px auto;
        max-width: 820px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65);
    }
    .splash-dcca {
        font-size: 0.98rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    .splash-title {
        font-size: 3.1rem;
        font-weight: 900;
        color: #F8FAFC;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .header-banner {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-left: 5px solid #E30613;
        border-radius: 12px;
        padding: 14px 22px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.30);
    }
    .brand-tag {
        font-size: 0.75rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .main-title {
        font-size: 1.52rem;
        font-weight: 900;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: 0.3px;
        line-height: 1.2;
    }
    .main-subtitle {
        font-size: 0.84rem;
        color: #94A3B8;
        margin-top: 3px;
    }
    .author-badge {
        background: rgba(227, 6, 19, 0.12);
        border: 1px solid rgba(227, 6, 19, 0.45);
        border-radius: 10px;
        padding: 9px 16px;
        text-align: right;
        font-weight: 700;
        color: #F8FAFC;
        font-size: 0.88rem;
    }
    .bag-showcase {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        gap: 14px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .predictor-main-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #F8FAFC;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .section-header {
        font-size: 0.93rem;
        font-weight: 700;
        color: #E2E8F0;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
        margin-top: 12px;
        margin-bottom: 10px;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.79rem !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #E30613 0%, #B8000C 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.0rem;
        padding: 11px 20px;
        box-shadow: 0 4px 15px rgba(227, 6, 19, 0.35);
    }
    .result-card {
        background: radial-gradient(circle at top, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .result-value {
        font-size: 3.1rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.05;
        margin: 4px 0;
    }
    .result-unit {
        font-size: 1.15rem;
        color: #E30613;
        font-weight: 700;
    }
    .conformal-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 5px 14px;
        border-radius: 20px;
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid rgba(14, 165, 233, 0.45);
        color: #E0F2FE;
        font-size: 0.80rem;
        font-weight: 600;
    }
    .real-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 4px 12px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.50);
        color: #ECFDF5;
        font-size: 0.84rem;
        font-weight: 700;
    }
    .kpi-mini {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #E30613;
        border-radius: 8px;
        padding: 9px 14px;
        margin-bottom: 8px;
    }
    .kpi-mini-title {
        font-size: 0.71rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .kpi-mini-val {
        font-size: 1.18rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 2px;
    }
    .metric-exec-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-top: 3px solid #E30613;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    }
    .metric-exec-title {
        font-size: 0.74rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.6px;
    }
    .metric-exec-val {
        font-size: 1.32rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .footer-disclaimer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 2.0rem;
        padding-top: 0.9rem;
        border-top: 1px solid #1E293B;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)

# PANTALLA DE PRESENTACIÓN INICIAL (PAUSADA Y EJECUTIVA)
if "splash_mostrado" not in st.session_state:
    placeholder = st.empty()
    logo_html = f'<div><img src="{LOGO_URI}" style="max-height:80px; margin-bottom:10px;"></div>' if LOGO_URI else ''
    bolsas_splash = "".join([
        f'<img src="{uri}" style="height:76px; margin:0 12px; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.5);">'
        for uri in BOLSAS_URI.values() if uri
    ])
    div_bolsas_splash = f'<div style="margin: 16px 0;">{bolsas_splash}</div>' if bolsas_splash else ''

    splash_html = (
        f'<div class="splash-box">'
        f'{logo_html}'
        f'<div class="splash-dcca">DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>'
        f'<div class="splash-title">DCC<span style="color:#E30613;">AI</span></div>'
        f'<div style="color:#E2E8F0; font-size:1.10rem; font-weight:600; margin-bottom:6px;">'
        f'Inteligencia Artificial para el Control de Calidad'
        f'</div>'
        f'<div style="color:#94A3B8; font-size:0.92rem; margin-bottom:14px;">'
        f'Modelamiento Predictivo de Resistencia a la Compresión'
        f'</div>'
        f'{div_bolsas_splash}'
        f'<div style="border-top:1px solid #334155; padding-top:15px; font-size:0.96rem; color:#F8FAFC;">'
        f'Elaborado por Ing. CIP Marco Antonio Uriel Martínez Sánchez'
        f'</div>'
        f'</div>'
    )

    with placeholder.container():
        st.markdown(splash_html, unsafe_allow_html=True)
        barra = st.progress(0, text="Inicializando arquitectura predictiva DCCAI...")
        for pct in range(0, 101, 20):
            time.sleep(0.80)
            barra.progress(pct, text="Sincronizando modelos de Machine Learning (Sol • Apu • Andino)...")
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
            "28d": ["modelo_GU_28d_PRO_produccion.pkl", "modelo_GU_28d_produccion.pkl", "modelo_M34_28d_produccion.pkl"]
        },
        "TI": {
            "1d": ["modelo_TI_1d_produccion.pkl"],
            "3d": ["modelo_TI_3d_produccion.pkl"],
            "7d": ["modelo_TI_7d_produccion.pkl"],
            "28d": ["modelo_TI_28d_PRO_produccion.pkl", "modelo_TI_28d_produccion.pkl"]
        },
        "HS": {
            "1d": ["modelo_HS_1d_produccion.pkl"],
            "3d": ["modelo_HS_3d_produccion.pkl"],
            "7d": ["modelo_HS_7d_produccion.pkl"],
            "28d": ["modelo_HS_28d_PRO_produccion.pkl", "modelo_HS_28d_produccion.pkl"]
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
# 4. LECTOR UNIVERSAL DE EXCEL Y MOTOR DE INFERENCIA INMUNE A VALUEERROR
# ==============================================================================
def normalizar_nombre(texto):
    if texto is None or pd.isna(texto):
        return ""
    return re.sub(
        r'[^a-z0-9]', '',
        unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()
    )

def leer_excel_universal(archivo_subido):
    bytes_data = archivo_subido.getvalue()
    nombre = archivo_subido.name.lower()

    for engine in [None, "openpyxl", "xlrd"]:
        try:
            return pd.read_excel(io.BytesIO(bytes_data), engine=engine)
        except Exception:
            pass
    try:
        tablas = pd.read_html(io.BytesIO(bytes_data))
        if tablas:
            return tablas[0]
    except Exception:
        pass
    try:
        return pd.read_csv(io.BytesIO(bytes_data), sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"No se pudo decodificar el archivo '{nombre}'. Detalle: {e}")

def formatear_valor_fecha(val):
    if val is None or pd.isna(val):
        return None
    if hasattr(val, "strftime"):
        return val.strftime("%d/%m/%y")
    txt = str(val).strip()
    if not txt or txt.lower() in ["nan", "nat", "none", ""]:
        return None
    txt = re.sub(r'\s+12:00\s*a\.?\s*m\.?', '', txt, flags=re.IGNORECASE)
    txt = re.sub(r'a\.?\s*m\.?', 'AM', txt, flags=re.IGNORECASE)
    txt = re.sub(r'p\.?\s*m\.?', 'PM', txt, flags=re.IGNORECASE)
    dt = pd.to_datetime(txt, errors='coerce')
    if pd.notna(dt):
        return dt.strftime("%d/%m/%y")
    return None

def a_serie_numerica(df_or_series):
    if isinstance(df_or_series, pd.DataFrame):
        s = df_or_series.iloc[:, 0]
    else:
        s = df_or_series
    return pd.to_numeric(s, errors='coerce')

def limpiar_dataframe_lotes(df_raw, features_esperadas, defaults_cem, edad_seleccionada=None):
    df = df_raw.copy()

    cols_norm_actual = [normalizar_nombre(c) for c in df.columns]
    if not any(k in cols_norm_actual for k in ['fecha', 'alita', 'blaine', 'cal', 'caliza']):
        for r_idx in range(min(5, len(df))):
            fila_vals = [normalizar_nombre(v) for v in df.iloc[r_idx].values if pd.notna(v)]
            if any(k in fila_vals for k in ['fecha', 'alita', 'blaine', 'cal', 'caliza']):
                df.columns = [str(v).strip() if pd.notna(v) else f"col_{k}" for k, v in enumerate(df.iloc[r_idx].values)]
                df = df.iloc[r_idx + 1:].reset_index(drop=True)
                break

    idx_corte = None
    for i in range(len(df)):
        textos_fila = [str(v).lower() for v in df.iloc[i].values if v is not None and not pd.isna(v)]
        fila_texto = " ".join(textos_fila)
        if any(p in fila_texto for p in ["promedio", "desviacion", "desviación", "estandar", "estándar", "average"]):
            idx_corte = i
            break
    if idx_corte is not None:
        df = df.iloc[:idx_corte].copy()

    idx_col_fecha = None
    for j, col_name in enumerate(df.columns):
        if normalizar_nombre(col_name) in ['fecha', 'date', 'fechamolienda']:
            idx_col_fecha = j
            break
    if idx_col_fecha is not None:
        df = df.iloc[:, idx_col_fecha:].copy()

    renombres = {}
    for col in df.columns:
        c_norm = normalizar_nombre(col)
        if c_norm in MAPEO_BD_EXACTO:
            renombres[col] = MAPEO_BD_EXACTO[c_norm]
        elif 'fecha' in c_norm:
            renombres[col] = 'Fecha'
    df = df.rename(columns=renombres)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if 'Fecha' in df.columns:
        fechas_limpias = [formatear_valor_fecha(v) for v in df['Fecha'].values]
        df['Fecha'] = fechas_limpias
        df = df[df['Fecha'].notna()].copy()
    else:
        cols_num = [c for c in features_esperadas if c in df.columns]
        if cols_num:
            for c in cols_num:
                df[c] = a_serie_numerica(df[c])
            df = df.dropna(subset=cols_num, how='all').copy()

    df_modelo = pd.DataFrame(index=df.index)
    cols_receta_fija = ['%Caliza', '%Yeso', '%Puzolana', '%Escoria']
    for col in features_esperadas:
        if col in df.columns:
            serie_num = a_serie_numerica(df[col])
            if col in cols_receta_fija:
                serie_num = serie_num.ffill().fillna(defaults_cem.get(col, 0.0))
            df_modelo[col] = serie_num
            df[col] = serie_num
        else:
            df_modelo[col] = defaults_cem.get(col, 0.0)

    columnas_vista = []
    if 'Fecha' in df.columns:
        columnas_vista.append('Fecha')
    columnas_vista.extend([c for c in features_esperadas if c in df.columns])

    if edad_seleccionada in ['1d', '3d', '7d', '28d']:
        target_activo = f"{edad_seleccionada[:-1]}_d"
        if target_activo in df.columns:
            df[target_activo] = a_serie_numerica(df[target_activo]).round(0).astype('Int64')
            columnas_vista.append(target_activo)

    df_vista_final = df[columnas_vista].reset_index(drop=True)
    return df_vista_final, df_modelo.reset_index(drop=True)

def extraer_columnas_internas_modelo(modelo, features_fallback):
    for cand in [modelo, getattr(modelo, "model_", None)]:
        if cand is None:
            continue
        if hasattr(cand, "feature_names_in_"):
            return list(cand.feature_names_in_)
        if hasattr(cand, "get_booster"):
            try:
                fn = cand.get_booster().feature_names
                if fn:
                    return list(fn)
            except Exception:
                pass
        if hasattr(cand, "booster_"):
            try:
                fn = cand.booster_.feature_name()
                if fn:
                    return list(fn)
            except Exception:
                pass
        if hasattr(cand, "steps"):
            ult = cand.steps[-1][1]
            if hasattr(ult, "get_booster"):
                try:
                    fn = ult.get_booster().feature_names
                    if fn:
                        return list(fn)
                except Exception:
                    pass
    return list(features_fallback)

def alinear_df_para_modelo(modelo, df_in, features_esperadas):
    cols_objetivo = extraer_columnas_internas_modelo(modelo, features_esperadas)
    mapa_norm = {normalizar_nombre(c): c for c in df_in.columns}
    df_out = pd.DataFrame(index=df_in.index)
    for col in cols_objetivo:
        c_n = normalizar_nombre(col)
        if col in df_in.columns:
            df_out[col] = a_serie_numerica(df_in[col]).fillna(-9999.0)
        elif c_n in mapa_norm:
            df_out[col] = a_serie_numerica(df_in[mapa_norm[c_n]]).fillna(-9999.0)
        else:
            df_out[col] = 0.0
    return df_out

def ejecutar_predict_seguro(modelo, df_in, features_esperadas):
    reparar_compatibilidad_sklearn(modelo)
    df_aln = alinear_df_para_modelo(modelo, df_in, features_esperadas)

    try:
        return np.asarray(modelo.predict(df_aln), dtype=float)
    except Exception:
        pass
    try:
        return np.asarray(modelo.predict(df_aln.to_numpy(dtype=float)), dtype=float)
    except Exception:
        pass

    X_arr = df_aln.to_numpy(dtype=float)
    est_final = modelo
    if hasattr(modelo, "steps"):
        for _, paso in modelo.steps[:-1]:
            if not isinstance(paso, SimpleImputer):
                try:
                    X_arr = paso.transform(X_arr)
                except Exception:
                    pass
        est_final = modelo.steps[-1][1]

    if hasattr(est_final, "get_booster"):
        return np.asarray(est_final.get_booster().inplace_predict(X_arr, validate_features=False), dtype=float)
    return np.asarray(est_final.predict(X_arr), dtype=float)

def predecir_con_modelo(modelo, df_features, features_esperadas, sigma_fallback=10.0):
    if hasattr(modelo, "predict_interval") and hasattr(modelo, "model_"):
        raw_pred = ejecutar_predict_seguro(modelo.model_, df_features, features_esperadas)
        y_hat = modelo.mu_y_ + modelo.factor_escala_ * (raw_pred - modelo.mu_pred_)
        q = getattr(modelo, "q_conformal_", 1.2816 * sigma_fallback)
        return y_hat, y_hat - q, y_hat + q
    else:
        y_hat = ejecutar_predict_seguro(modelo, df_features, features_esperadas)
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
logo_header_html = f'<img src="{LOGO_URI}" style="height:48px; margin-right:16px;">' if LOGO_URI else ''

header_html = (
    f'<div class="header-banner">'
    f'<div style="display:flex; align-items:center;">'
    f'{logo_header_html}'
    f'<div>'
    f'<div class="brand-tag">DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>'
    f'<div class="main-title">DCC<span style="color:#E30613;">AI</span>: Inteligencia Artificial para el Control de Calidad</div>'
    f'<div class="main-subtitle">Modelamiento Predictivo de Resistencia a la Compresión</div>'
    f'</div>'
    f'</div>'
    f'<div class="author-badge">Elaborado por Ing. CIP Marco Antonio Uriel Martínez Sánchez</div>'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

col_sel1, col_sel2, col_bag = st.columns([1.5, 1.5, 1.2])
with col_sel1:
    tipo_cemento_label = st.selectbox("Cemento", list(CONFIG_CEMENTOS.keys()), index=0)
with col_sel2:
    edad_label = st.selectbox(
        "Edad de Predicción",
        [
            "1 día",
            "3 días",
            "7 días",
            "28 días",
            "Evolución multiedad (1, 3, 7 y 28 días)"
        ],
        index=3
    )

cfg = CONFIG_CEMENTOS[tipo_cemento_label]
cod_cem = cfg["codigo"]

with col_bag:
    bag_uri = BOLSAS_URI.get(cod_cem)
    img_bag_top = f'<img src="{bag_uri}" style="height:56px; border-radius:4px;">' if bag_uri else ''
    st.markdown(
        f'<div class="bag-showcase">{img_bag_top}'
        f'<div><div style="font-size:0.71rem; color:#E30613; font-weight:800; text-transform:uppercase;">{cfg["norma"]}</div>'
        f'<div style="font-size:0.94rem; color:#F8FAFC; font-weight:800;">{cfg["marca"]}</div></div></div>',
        unsafe_allow_html=True
    )


# ==============================================================================
# 6. PESTAÑAS EJECUTIVAS DE CONTROL DE CALIDAD (ORTOGRAFÍA RAE)
# ==============================================================================
tab_sim, tab_batch, tab_spc = st.tabs([
    "🎛️ Inferencia y Variables Predictoras",
    "📂 Carga por Lotes y Métricas",
    "📈 Diagnóstico de Residuales y Control Estadístico"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: INFERENCIA Y VARIABLES PREDICTORAS
# ------------------------------------------------------------------------------
with tab_sim:
    defaults = cfg["defaults"].copy()
    features_activas = cfg["features"]
    valor_real_lote = None

    if f"lotes_{cod_cem}" in st.session_state:
        df_mem_vista, df_mem_mod = st.session_state[f"lotes_{cod_cem}"]
        if len(df_mem_vista) > 0:
            opciones_lote = ["Simulación interactiva libre"] + [
                f"Lote {i+1} | Fecha: {df_mem_vista['Fecha'].iloc[i]}" if 'Fecha' in df_mem_vista.columns else f"Lote #{i+1}"
                for i in range(len(df_mem_vista))
            ]
            lote_elegido = st.selectbox("🔍 Sincronizar variables y Valor Real desde un lote del Excel cargado:", opciones_lote)
            if lote_elegido != "Simulación interactiva libre":
                idx_lote = opciones_lote.index(lote_elegido) - 1
                for f_col in features_activas:
                    if f_col in df_mem_mod.columns and pd.notna(df_mem_mod[f_col].iloc[idx_lote]):
                        defaults[f_col] = float(df_mem_mod[f_col].iloc[idx_lote])
                mapa_ed_col = {"1 día": "1_d", "3 días": "3_d", "7 días": "7_d", "28 días": "28_d"}
                col_real_buscada = mapa_ed_col.get(edad_label)
                if col_real_buscada and col_real_buscada in df_mem_vista.columns:
                    val_r = pd.to_numeric(df_mem_vista[col_real_buscada].iloc[idx_lote], errors='coerce')
                    if pd.notna(val_r):
                        valor_real_lote = int(round(float(val_r)))

    st.markdown(
        f'<div class="predictor-main-title">Variables Predictoras: <span style="color:#E30613;">{cfg["marca"]}</span></div>',
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

    registro_actual = {
        '%Caliza': caliza, '%Yeso': yeso, '%Puzolana': puzolana, '%Escoria': escoria,
        'M325': m325, 'Blaine': blaine, 'SO3': so3, 'Álcali_Eq': alcali_eq,
        'Alita': alita, 'Belita': belita, 'Cal_Libre': cal_libre,
        'C3A_Cúbico': c3a_cub, 'C3A_Ortorómbico': c3a_orto,
        'Gypsum': gypsum, 'Hemidrato': hemidrato, 'Anhidrita': anhidrita,
        'Agua_Cemento': agua_cemento, 'Fluidez': fluidez
    }
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

    estilo_multiedad = "📈 Curva Cinética Continua"
    if "multiedad" in edad_label:
        estilo_multiedad = st.radio(
            "Visualización de desarrollo de resistencia:",
            ["📈 Curva Cinética Continua", "📊 Gráfico de Barras Comparativo"],
            horizontal=True
        )

    if st.button("Ejecutar Predicción", use_container_width=True):
        mapa_edad = {"1 día": "1d", "3 días": "3d", "7 días": "7d", "28 días": "28d"}
        edad_elegida = mapa_edad.get(edad_label, "MULTI")

        if edad_elegida != "MULTI":
            mod_obj, _ = REPO_MODELOS[cod_cem][edad_elegida]
            sigma_fb = cfg["incertidumbre_base"][edad_elegida]

            if mod_obj is not None:
                y_p, p10, p90 = predecir_con_modelo(mod_obj, df_input_single, features_activas, sigma_fallback=sigma_fb)
                val_c = int(round(float(y_p[0])))
                val_p10 = int(round(float(p10[0])))
                val_p90 = int(round(float(p90[0])))

                html_real = ""
                if valor_real_lote is not None:
                    desvio = abs(valor_real_lote - val_c)
                    html_real = f'<div><span class="real-badge">✔️ Valor Real: {valor_real_lote} kg/cm² (Error: {desvio} kg/cm²)</span></div>'

                bag_card_html = f'<div style="margin-bottom:8px;"><img src="{bag_uri}" style="height:62px; border-radius:4px;"></div>' if bag_uri else ''

                card_html = (
                    f'<div class="result-card">'
                    f'{bag_card_html}'
                    f'<div style="font-size:0.80rem; color:#94A3B8; font-weight:700; margin-bottom:6px;">'
                    f'{tipo_cemento_label} &bull; {edad_label}'
                    f'</div>'
                    f'<div class="result-value">{val_c} <span class="result-unit">kg/cm²</span></div>'
                    f'{html_real}'
                    f'<div><span class="conformal-badge">'
                    f'🛡️ Intervalo Conformal (P10 – P90): <b>{val_p10}</b> a <b>{val_p90} kg/cm²</b>'
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
                        title="Sensibilidad en la Varianza Operativa (Feature Importance)",
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Peso Relativo en la Variabilidad del Proceso (%)", showgrid=False),
                        yaxis=dict(showgrid=False),
                        height=275, margin=dict(l=10, r=10, t=35, b=10)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.caption("ℹ️ *Nota técnica: Refleja qué parámetros fisicoquímicos gobiernan las variaciones día a día dentro del rango operativo de planta.*")
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
                    vc = int(round(float(yp[0])))
                    vp10 = int(round(float(l10[0])))
                    vp90 = int(round(float(l90[0])))
                    preds_c.append(vc)
                    preds_p10.append(vp10)
                    preds_p90.append(vp90)
                    mini_card_html = (
                        f'<div class="result-card" style="padding:14px;">'
                        f'<div style="font-size:0.80rem; color:#94A3B8; font-weight:700;">{etiquetas_ed[idx]}</div>'
                        f'<div class="result-value" style="font-size:2.15rem;">{vc} <span class="result-unit" style="font-size:0.92rem;">kg/cm²</span></div>'
                        f'<div style="font-size:0.73rem; color:#94A3B8; margin-top:4px;">P10–P90: [{vp10} – {vp90}]</div>'
                        f'</div>'
                    )
                    cols_cards[idx].markdown(mini_card_html, unsafe_allow_html=True)
                else:
                    preds_c.append(np.nan)
                    preds_p10.append(np.nan)
                    preds_p90.append(np.nan)

            if not all(np.isnan(preds_c)):
                fig_curva = go.Figure()
                if "Curva" in estilo_multiedad:
                    fig_curva.add_trace(go.Scatter(
                        x=dias_num + dias_num[::-1],
                        y=preds_p90 + preds_p10[::-1],
                        fill='toself', fillcolor='rgba(227, 6, 19, 0.20)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo Conformal P10–P90',
                        hoverinfo='skip'
                    ))
                    fig_curva.add_trace(go.Scatter(
                        x=dias_num, y=preds_c, mode='lines+markers+text',
                        line=dict(color='#E30613', width=3),
                        marker=dict(size=10, color='#F8FAFC', line=dict(width=2, color='#E30613')),
                        text=[f"{int(v)}" if not np.isnan(v) else "" for v in preds_c],
                        textposition="top center", name='Evolución Predicha',
                        hovertemplate="Edad: %{x} días<br>Predicción: %{y:d} kg/cm²<extra></extra>"
                    ))
                    fig_curva.update_layout(
                        title=f"Curva de Desarrollo de Resistencia (1, 3, 7 y 28 días): {cfg['marca']}",
                        xaxis=dict(
                            title="Edad de Ensayo",
                            tickmode="array",
                            tickvals=[1, 3, 7, 28],
                            ticktext=["1 día", "3 días", "7 días", "28 días"],
                            showgrid=False
                        ),
                        yaxis=dict(title="Resistencia (kg/cm²)", tickformat="d", showgrid=False),
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=340
                    )
                else:
                    err_sup = [p90 - c for c, p90 in zip(preds_c, preds_p90)]
                    err_inf = [c - p10 for c, p10 in zip(preds_c, preds_p10)]
                    fig_curva.add_trace(go.Bar(
                        x=etiquetas_ed, y=preds_c,
                        marker_color=['#334155', '#475569', '#B8000C', '#E30613'],
                        text=[f"{int(v)} kg/cm²" for v in preds_c],
                        textposition='outside',
                        error_y=dict(type='data', symmetric=False, array=err_sup, arrayminus=err_inf, color='#F8FAFC', thickness=1.5),
                        name='Resistencia Predicha',
                        hovertemplate="Edad: %{x}<br>Predicción: %{y:d} kg/cm²<extra></extra>"
                    ))
                    fig_curva.update_layout(
                        title=f"Desarrollo Comparativo de Resistencia por Edad (1, 3, 7 y 28 días): {cfg['marca']}",
                        xaxis=dict(title="Edad de Ensayo", showgrid=False),
                        yaxis=dict(title="Resistencia (kg/cm²)", tickformat="d", range=[0, max(preds_p90) * 1.15], showgrid=False),
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=340
                    )
                st.plotly_chart(fig_curva, use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 2: CARGA POR LOTES Y MÉTRICAS (3 KPIs COMPACTOS AL FINAL)
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown("##### 📂 Carga por Lotes y Resultado de Métricas")

    mapa_horizonte_batch = {
        "1 día": "1d",
        "3 días": "3d",
        "7 días": "7d",
        "28 días": "28d"
    }
    edad_batch = mapa_horizonte_batch.get(edad_label, "28d")
    target_col = f"{edad_batch[:-1]}_d"

    archivo_subido = st.file_uploader(
        f"Sube el archivo Excel exportado de la base de datos ({tipo_cemento_label})",
        type=["xlsx", "xls", "xlsm", "csv"]
    )

    if archivo_subido is not None:
        try:
            df_raw = leer_excel_universal(archivo_subido)
            df_vista, df_mod = limpiar_dataframe_lotes(
                df_raw, cfg["features"], cfg["defaults"], edad_seleccionada=edad_batch
            )
            st.session_state[f"lotes_{cod_cem}"] = (df_vista, df_mod)

            mod_b, _ = REPO_MODELOS[cod_cem][edad_batch]
            if mod_b is not None and len(df_mod) > 0:
                y_b, p10_b, p90_b = predecir_con_modelo(
                    mod_b, df_mod, cfg["features"], sigma_fallback=cfg["incertidumbre_base"][edad_batch]
                )
                y_b_int = np.round(y_b).astype(int)
                df_out = df_vista.copy()
                df_out.insert(0, f"{edad_batch}_PREDICTION", y_b_int)

                mae_g, mape_g, rmse_g = None, None, None

                if target_col in df_out.columns:
                    y_real_b = pd.to_numeric(df_out[target_col], errors='coerce').round(0)
                    mask_b = y_real_b.notna()
                    n_reales = int(mask_b.sum())

                    df_out["MAE"] = np.where(mask_b, np.abs(y_real_b - y_b_int), np.nan)
                    df_out["MAPE (%)"] = np.where(mask_b, np.round((df_out["MAE"] / y_real_b) * 100, 2), np.nan)
                    df_out["RMSE"] = np.where(mask_b, np.round((y_real_b - y_b_int) ** 2, 0), np.nan)

                    if n_reales > 0:
                        y_r_valid = y_real_b[mask_b].astype(int)
                        y_p_valid = y_b_int[mask_b].astype(int)
                        fechas_valid = df_out.loc[mask_b, 'Fecha'].values if 'Fecha' in df_out.columns else np.arange(1, n_reales + 1)

                        mae_g = mean_absolute_error(y_r_valid, y_p_valid)
                        mape_g = np.mean(np.abs((y_r_valid - y_p_valid) / y_r_valid)) * 100
                        rmse_g = np.sqrt(mean_squared_error(y_r_valid, y_p_valid))
                        st.session_state["ultimo_df_validado"] = (df_out, y_r_valid, y_p_valid, fechas_valid, edad_batch)

                st.dataframe(df_out, use_container_width=True, height=370)

                if mae_g is not None:
                    st.markdown('<div class="section-header">📊 Resumen Estadístico de Desempeño del Modelo</div>', unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(
                        f'<div class="metric-exec-card"><div class="metric-exec-title">MAE Promedio</div>'
                        f'<div class="metric-exec-val">{mae_g:.1f} kg/cm²</div></div>',
                        unsafe_allow_html=True
                    )
                    m2.markdown(
                        f'<div class="metric-exec-card"><div class="metric-exec-title">MAPE Promedio</div>'
                        f'<div class="metric-exec-val">{mape_g:.2f}%</div></div>',
                        unsafe_allow_html=True
                    )
                    m3.markdown(
                        f'<div class="metric-exec-card"><div class="metric-exec-title">RMSE Global</div>'
                        f'<div class="metric-exec-val">{rmse_g:.1f} kg/cm²</div></div>',
                        unsafe_allow_html=True
                    )

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
            elif len(df_mod) == 0:
                st.warning("El archivo fue leído pero no se detectaron filas válidas con las variables del cemento.")
            else:
                st.error(f"No se pudo cargar el archivo `.pkl` de **{cod_cem} - {edad_batch}**.")
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")

# ------------------------------------------------------------------------------
# PESTAÑA 3: 3 GRÁFICOS SIN CUADRÍCULA DE FONDO (SHOWGRID=FALSE)
# ------------------------------------------------------------------------------
with tab_spc:
    st.markdown("##### 📈 Diagnóstico de Residuales y Control Estadístico de Calidad")
    if "ultimo_df_validado" in st.session_state:
        df_v, y_r, y_p, fechas_v, ed_v = st.session_state["ultimo_df_validado"]
        y_r_int = np.round(y_r).astype(int)
        y_p_int = np.round(y_p).astype(int)
        residuales_int = (y_p_int - y_r_int).astype(int)

        # GRÁFICO 1: SEGUIMIENTO TEMPORAL LIMPIO (SIN LÍNEAS HORIZONTALES DE FONDO)
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=fechas_v, y=y_r_int,
            mode='markers',
            name='Dato Real (Prensa)',
            marker=dict(symbol='x', size=10, color='#38BDF8', line=dict(width=2, color='#38BDF8')),
            hovertemplate="Fecha: %{x}<br>Dato Real: %{y:d} kg/cm²<extra></extra>"
        ))
        fig_time.add_trace(go.Scatter(
            x=fechas_v, y=y_p_int,
            mode='markers',
            name='Dato Predicho (DCCAI)',
            marker=dict(symbol='circle', size=9, color='#E30613', line=dict(width=1, color='#F8FAFC')),
            hovertemplate="Fecha: %{x}<br>Dato Predicho: %{y:d} kg/cm²<extra></extra>"
        ))
        fig_time.update_layout(
            title=f"Seguimiento Temporal de Resistencias por Fecha ({cod_cem} - {ed_v})",
            xaxis=dict(title="Fecha de Molienda", type="category", showgrid=False),
            yaxis=dict(title="Resistencia (kg/cm²)", tickformat="d", showgrid=False, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # GRÁFICOS 2 Y 3: PREDICCIÓN VS REAL Y RESIDUALES (SIN CUADRÍCULA DE FONDO)
        g1, g2 = st.columns(2)
        with g1:
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=y_r_int, y=y_p_int,
                mode='markers',
                name='Lotes Evaluados',
                marker=dict(symbol='circle', size=9, color='#38BDF8', line=dict(width=1, color='#F8FAFC')),
                hovertemplate="Valor Real: %{x:d} kg/cm²<br>Predicción DCCAI: %{y:d} kg/cm²<extra></extra>"
            ))
            fig_sc.update_layout(
                title=f"Predicción vs Valor Real ({cod_cem} - {ed_v})",
                xaxis=dict(title="Valor Real (kg/cm²)", tickformat="d", showgrid=False, zeroline=False),
                yaxis=dict(title="Predicción DCCAI (kg/cm²)", tickformat="d", showgrid=False, zeroline=False),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=330
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with g2:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_r_int, y=residuales_int,
                mode='markers',
                name='Residuales',
                marker=dict(symbol='circle', size=9, color='#E30613', line=dict(width=1, color='#F8FAFC')),
                hovertemplate="Valor Real: %{x:d} kg/cm²<br>Residual: %{y:+d} kg/cm²<extra></extra>"
            ))
            fig_res.add_hline(y=0, line_color="#64748B", line_dash="dash", line_width=1)
            fig_res.update_layout(
                title=f"Distribución de Residuales ({cod_cem} - {ed_v})",
                xaxis=dict(title="Valor Real (kg/cm²)", tickformat="d", showgrid=False, zeroline=False),
                yaxis=dict(title="Residual (Predicho - Real) [kg/cm²]", tickformat="d", showgrid=False, zeroline=False),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=330
            )
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Carga un archivo en la pestaña **Carga por Lotes y Métricas** para visualizar el seguimiento temporal por fecha y el diagnóstico de residuales.")

# ==============================================================================
# 7. PIE DE PÁGINA INSTITUCIONAL
# ==============================================================================
footer_html = (
    '<div class="footer-disclaimer">'
    '<div><b>UNACEM: DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO (DCCAI)</b> &nbsp;|&nbsp; '
    'Elaborado por Ing. CIP Marco Antonio Uriel Martínez Sánchez</div>'
    '<div>Las estimaciones e intervalos conformales son herramientas de soporte operativo y pueden presentar variaciones de acuerdo al proceso.</div>'
    '</div>'
)
st.markdown(footer_html, unsafe_allow_html=True)
