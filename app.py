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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 0. CLASES PERSONALIZADAS Y REGISTRO EN __main__ PARA STREAMLIT CLOUD
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
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS REFINADOS
# ==============================================================================
st.set_page_config(
    page_title="DCCAI | UNACEM - DCCA Atocongo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def obtener_logo_b64(ruta_logo="unacem_logo.png"):
    if os.path.exists(ruta_logo):
        with open(ruta_logo, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

LOGO_B64 = obtener_logo_b64()

st.markdown("""
<style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 98% !important;
    }
    .splash-box {
        background: radial-gradient(circle at center, #1f242e 0%, #0d0f14 100%);
        border: 1px solid #303642;
        border-top: 4px solid #E30613;
        border-radius: 16px;
        padding: 45px 25px;
        text-align: center;
        margin: 30px auto;
        max-width: 760px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.65);
    }
    .splash-dcca {
        font-size: 1.05rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    .splash-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .header-banner {
        background: linear-gradient(90deg, #12151c 0%, #1a1f29 100%);
        border: 1px solid #2d3340;
        border-left: 5px solid #E30613;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: nowrap;
        gap: 15px;
    }
    .brand-tag {
        font-size: 0.78rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 1.4px;
        text-transform: uppercase;
    }
    .main-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.15;
    }
    .author-badge {
        background: rgba(227, 6, 19, 0.1);
        border: 1px solid rgba(227, 6, 19, 0.35);
        border-radius: 8px;
        padding: 6px 12px;
        text-align: right;
        min-width: 280px;
        flex-shrink: 0;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f3f4f6;
        border-bottom: 1px solid #303642;
        padding-bottom: 6px;
        margin-top: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
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
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.02rem;
        padding: 10px 18px;
        box-shadow: 0 4px 14px rgba(227, 6, 19, 0.25);
    }
    .result-card {
        background: radial-gradient(circle at top, #1f242e 0%, #11141a 100%);
        border: 1px solid #303642;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 8px;
    }
    .result-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #f9fafb;
        line-height: 1;
    }
    .result-unit {
        font-size: 1.25rem;
        color: #E30613;
        font-weight: 700;
    }
    .conformal-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(227, 6, 19, 0.14);
        border: 1px solid rgba(227, 6, 19, 0.4);
        color: #f3f4f6;
        font-size: 0.81rem;
        font-weight: 600;
    }
    .kpi-mini {
        background: #161922;
        border-left: 3px solid #E30613;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .kpi-mini-title {
        font-size: 0.72rem;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .kpi-mini-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }
    .footer-disclaimer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 1.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #1f242e;
    }
</style>
""", unsafe_allow_html=True)

# PANTALLA DE PRESENTACIÓN INICIAL (SPLASH SCREEN)
if "splash_mostrado" not in st.session_state:
    placeholder = st.empty()
    logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="max-height:75px; margin-bottom:8px;">' if LOGO_B64 else '<div style="font-size:2.5rem;">🏭</div>'
    with placeholder.container():
        st.markdown(f"""
        <div class="splash-box">
            {logo_html}
            <div class="splash-dcca">UNACEM &bull; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>
            <div class="splash-title">DCCAI | QuAI &mdash; M.A.R.C.O. v2.0</div>
            <div style="color:#9ca3af; font-size:0.92rem; margin-bottom:14px;">
                Sistema Inteligente de Predicción de Resistencia a la Compresión y Auditoría Quimiométrica
            </div>
            <div style="border-top:1px solid #2d3340; padding-top:14px; font-size:0.98rem; color:#e5e7eb;">
                <b>Elaborado por:</b> Ing. Marco Antonio Uriel Martínez Sánchez<br>
                <span style="font-size:0.82rem; color:#9ca3af;">(Martínez Sánchez, Marco Antonio Uriel)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        barra = st.progress(0, text="Iniciando motores quimiométricos...")
        for pct in range(0, 101, 33):
            time.sleep(0.35)
            barra.progress(pct, text="Cargando modelos predictivos y normalizadores...")
    placeholder.empty()
    st.session_state["splash_mostrado"] = True


# ==============================================================================
# 2. CATÁLOGO QUIMIOMÉTRICO Y DICCIONARIO DE MAPEO DE COLUMNAS (BD A MODELO)
# ==============================================================================
CONFIG_CEMENTOS = {
    "Cemento Tipo GU (16 Variables)": {
        "codigo": "GU",
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
        "rangos_3sigma": {
            '%Caliza': (15.0, 26.0), 'Blaine': (3300.0, 4400.0),
            'SO3': (2.1, 3.5), 'Alita': (38.0, 54.0), 'Agua_Cemento': (46.0, 53.0)
        },
        "incertidumbre_base": {"1d": 6.5, "3d": 9.0, "7d": 11.0, "28d": 14.0}
    },
    "Cemento Tipo I (15 Variables)": {
        "codigo": "TI",
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
        "rangos_3sigma": {
            '%Caliza': (1.0, 7.5), 'Blaine': (3200.0, 4150.0),
            'SO3': (2.0, 3.4), 'Alita': (48.0, 64.0)
        },
        "incertidumbre_base": {"1d": 7.0, "3d": 10.5, "7d": 12.0, "28d": 15.0}
    },
    "Cemento Tipo HS (18 Variables)": {
        "codigo": "HS",
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
        "rangos_3sigma": {
            '%Caliza': (8.0, 12.0), '%Puzolana': (11.0, 17.5), '%Escoria': (15.0, 21.0),
            'Blaine': (4600.0, 5350.0), 'SO3': (2.1, 2.9), 'Alita': (40.0, 51.0),
            'Agua_Cemento': (47.5, 51.0)
        },
        "incertidumbre_base": {"1d": 6.0, "3d": 9.8, "7d": 11.5, "28d": 14.1}
    }
}

# Diccionario universal de mapeo para normalizar nombres en bruto de bases de datos
MAPEO_COLUMNAS_CRUDAS = {
    'caliza': '%Caliza', 'porcentajecaliza': '%Caliza', '%caliza': '%Caliza',
    'yeso': '%Yeso', 'porcentajeyeso': '%Yeso', '%yeso': '%Yeso',
    'puzolana': '%Puzolana', '%puzolana': '%Puzolana',
    'escoria': '%Escoria', '%escoria': '%Escoria',
    'm325': 'M325', '%m325': 'M325',
    'blaine': 'Blaine',
    'so3': 'SO3', 'sulfurtoxide': 'SO3',
    'alcalieq': 'Álcali_Eq', 'alcaliequivalente': 'Álcali_Eq', 'alkali': 'Álcali_Eq', 'alcali_eq': 'Álcali_Eq',
    'alita': 'Alita', 'c3s': 'Alita',
    'belita': 'Belita', 'c2s': 'Belita',
    'callibre': 'Cal_Libre', 'calibre': 'Cal_Libre', 'cao_libre': 'Cal_Libre',
    'c3acubico': 'C3A_Cúbico', 'c3acub': 'C3A_Cúbico', 'c3a_cubico': 'C3A_Cúbico',
    'c3aortorrombico': 'C3A_Ortorómbico', 'c3aorto': 'C3A_Ortorómbico', 'c3a_ortorrombico': 'C3A_Ortorómbico',
    'gypsum': 'Gypsum',
    'hemidrato': 'Hemidrato', 'hemihidrato': 'Hemidrato',
    'anhidrita': 'Anhidrita',
    'aguacemento': 'Agua_Cemento', 'a/c': 'Agua_Cemento', 'agua_cemento': 'Agua_Cemento',
    'fluidez': 'Fluidez'
}


# ==============================================================================
# 3. CARGA ROBUSTA DE MODELOS (.PKL)
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
    errores_carga = {}
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
                    except Exception as e:
                        errores_carga[r] = str(e)
            repo[cem][edad] = (modelo_cargado, ruta_activa)
    return repo, errores_carga

REPO_MODELOS, ERRORES_PKL = cargar_repositorio_modelos()


# ==============================================================================
# 4. MOTOR DE INFERENCIA Y ALINEACIÓN DE DATOS CRUDOS
# ==============================================================================
def normalizar_nombre(texto):
    return re.sub(
        r'[^a-z0-9]', '',
        unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()
    )

def limpiar_y_mapear_dataframe_crudo(df_input, features_esperadas):
    """Limpia filas de promedios/totales y mapea nombres crudos de BD al modelo."""
    df = df_input.copy()
    
    # 1. Eliminar filas de resumen o promedios al fondo de la base de datos
    if len(df) > 0:
        col_primera = str(df.columns[0])
        mask_basura = df[col_primera].astype(str).str.lower().str.contains('promedio|average|media|total|sum', na=False)
        df = df[~mask_basura]

    # 2. Mapear nombres de columnas crudas
    nuevos_nombres = {}
    for col in df.columns:
        c_norm = normalizar_nombre(col)
        if c_norm in MAPEO_COLUMNAS_CRUDAS:
            nuevos_nombres[col] = MAPEO_COLUMNAS_CRUDAS[c_norm]
        else:
            # Búsqueda parcial por si hay sufijos
            encontrado = False
            for clave_map, col_real in MAPEO_COLUMNAS_CRUDAS.items():
                if clave_map in c_norm:
                    nuevos_nombres[col] = col_real
                    encontrado = True
                    break
            if not encontrado:
                nuevos_nombres[col] = col
                
    df = df.rename(columns=nuevos_nombres)

    # 3. Alinear asegurando que existan todas las features esperadas
    df_alineado = pd.DataFrame(index=df.index)
    for col in features_esperadas:
        if col in df.columns:
            df_alineado[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df_alineado[col] = 0.0 # Columnas faltantes por default
            
    # Mantener columnas auxiliares como Fecha o targets si existen
    for col_aux in ['Fecha', 'fecha', '1_d', '3_d', '7_d', '28_d']:
        if col_aux in df.columns:
            df_alineado[col_aux] = df[col_aux]
            
    return df_alineado

def ejecutar_predict_seguro(modelo, df_aln):
    reparar_compatibilidad_sklearn(modelo)
    try:
        return np.asarray(modelo.predict(df_aln), dtype=float)
    except AttributeError:
        X_val = df_aln.select_dtypes(include=[np.number]).fillna(-9999.0).values
        if hasattr(modelo, "steps"):
            for nombre_paso, paso in modelo.steps[:-1]:
                if not isinstance(paso, SimpleImputer):
                    X_val = paso.transform(X_val)
            return np.asarray(modelo.steps[-1][1].predict(X_val), dtype=float)
        raise

def predecir_con_modelo(modelo, df_input, features_esperadas, sigma_fallback=10.0):
    df_aln = limpiar_y_mapear_dataframe_crudo(df_input, features_esperadas)
    if hasattr(modelo, "predict_interval"):
        try:
            reparar_compatibilidad_sklearn(modelo)
            y_hat, p10, p90 = modelo.predict_interval(df_aln[features_esperadas])
            return np.asarray(y_hat, dtype=float), np.asarray(p10, dtype=float), np.asarray(p90, dtype=float)
        except AttributeError:
            raw_pred = ejecutar_predict_seguro(modelo.model_, df_aln[features_esperadas])
            y_hat = modelo.mu_y_ + modelo.factor_escala_ * (raw_pred - modelo.mu_pred_)
            q = getattr(modelo, "q_conformal_", 1.2816 * sigma_fallback)
            return y_hat, y_hat - q, y_hat + q
    else:
        y_hat = ejecutar_predict_seguro(modelo, df_aln[features_esperadas])
        q_80 = 1.2816 * sigma_fallback
        return y_hat, y_hat - q_80, y_hat + q_80

def calcular_contribucion_local_kg(modelo, df_fila, defaults_dict, features_esperadas):
    df_base = pd.DataFrame([{k: defaults_dict[k] for k in features_esperadas}])
    pred_base, _, _ = predecir_con_modelo(modelo, df_base, features_esperadas)
    val_base = float(pred_base[0])

    impactos = {}
    for col in features_esperadas:
        df_temp = df_base.copy()
        df_temp[col] = df_fila[col].values[0]
        pred_mod, _, _ = predecir_con_modelo(modelo, df_temp, features_esperadas)
        impactos[col] = float(pred_mod[0]) - val_base
    return val_base, impactos


# ==============================================================================
# 5. BANNER INSTITUCIONAL PERMANENTE
# ==============================================================================
logo_header_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:48px; margin-right:14px;">' if LOGO_B64 else ''

st.markdown(f"""
<div class="header-banner">
    <div style="display:flex; align-items:center;">
        {logo_header_html}
        <div>
            <div class="brand-tag">UNACEM &bull; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>
            <div class="main-title">DCCAI | QuAI &mdash; M.A.R.C.O. v2.0</div>
        </div>
    </div>
    <div class="author-badge">
        <div style="font-size:0.68rem; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Elaborado por</div>
        <div style="font-weight:700; color:#ffffff; font-size:0.86rem;">Ing. Marco Antonio Uriel Martínez Sánchez</div>
        <div style="font-size:0.72rem; color:#d1d5db;">Martínez Sánchez, Marco Antonio Uriel</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sel1, col_sel2, col_sel3 = st.columns([1.4, 1.4, 1.2])
with col_sel1:
    tipo_cemento_label = st.selectbox("Familia de Cemento", list(CONFIG_CEMENTOS.keys()), index=0)
with col_sel2:
    horizonte = st.selectbox(
        "Horizonte de Predicción",
        [
            "1 Día (Resistencia Temprana)",
            "3 Días (Control Operativo)",
            "7 Días (Desempeño Intermedio)",
            "28 Días (Certificación Final)",
            "Vista Multi-Edad Completa (1, 3, 7 y 28 Días)"
        ],
        index=3
    )
with col_sel3:
    cfg = CONFIG_CEMENTOS[tipo_cemento_label]
    cod_cem = cfg["codigo"]
    usar_cascada_28d = st.toggle("Activar Modo Cascada 28d", value=False)


# ==============================================================================
# 6. PESTAÑAS DE OPERACIÓN INDUSTRIAL
# ==============================================================================
tab_sim, tab_batch, tab_spc = st.tabs([
    "🎛️ Simulador Quimiométrico & Explicabilidad SHAP",
    "📂 Carga Masiva Excel & Evaluación Ciega (MAE / MAPE / RMSE)",
    "📈 Diagnóstico de Residuales & Control Estadístico (ASTM C109)"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: SIMULADOR AGRUPADO POR SECCIONES TÉCNICAS
# ------------------------------------------------------------------------------
with tab_sim:
    defaults = cfg["defaults"]
    features_activas = cfg["features"]

    # SECCIÓN 1: DIFRACCIÓN DE RAYOS X (XRD RIETVELD)
    st.markdown('<div class="section-header">🔬 1. Difracción de Rayos X (XRD Rietveld)</div>', unsafe_allow_html=True)
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

    # SECCIÓN 2: FLUORESCENCIA DE RAYOS X & DOSIFICACIÓN
    st.markdown('<div class="section-header">⚗️ 2. Fluorescencia de Rayos X & Dosificación</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        caliza = st.number_input("%Caliza", value=float(defaults['%Caliza']), step=0.10, format="%.2f")
        so3 = st.number_input("SO3 (%)", value=float(defaults['SO3']), step=0.05, format="%.2f")
    with f2:
        yeso = st.number_input("%Yeso", value=float(defaults['%Yeso']), step=0.10, format="%.2f")
        alcali_eq = st.number_input("Álcali_Eq", value=float(defaults['Álcali_Eq']), step=0.01, format="%.2f")
    with f3:
        if cod_cem == "HS":
            puzolana = st.number_input("%Puzolana", value=float(defaults['%Puzolana']), step=0.20, format="%.2f")
        else:
            puzolana = 0.0
            st.write("")
    with f4:
        if cod_cem == "HS":
            escoria = st.number_input("%Escoria", value=float(defaults['%Escoria']), step=0.20, format="%.2f")
        else:
            escoria = 0.0
            st.write("")

    # SECCIÓN 3: CONTROL DE PROCESOS Y ENSAYOS FÍSICOS (M325 & Blaine)
    st.markdown('<div class="section-header">⚙️ 3. Ensayos de Control de Procesos y Físicos (M325 & Blaine)</div>', unsafe_allow_html=True)
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
    with p4:
        fluidez = st.number_input("Fluidez (%)", value=float(defaults['Fluidez']), step=1.0, format="%.1f")

    res_1d_real, res_3d_real = None, None
    if usar_cascada_28d:
        cc1, cc2 = st.columns(2)
        with cc1:
            res_1d_real = st.number_input("Resistencia Real 1 Día (kg/cm²)", value=140.0, step=1.0)
        with cc2:
            res_3d_real = st.number_input("Resistencia Real 3 Días (kg/cm²)", value=252.0, step=1.0)

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

    if st.button("Calcular Predicción e Intervalo Conformal", use_container_width=True):
        mapa_edad = {"1 Día": "1d", "3 Días": "3d", "7 Días": "7d", "28 Días": "28d"}
        edad_elegida = next((v for k, v in mapa_edad.items() if k in horizonte), "MULTI")

        if edad_elegida != "MULTI":
            clave_m = "28d_CASCADA" if (edad_elegida == "28d" and usar_cascada_28d and REPO_MODELOS[cod_cem]["28d_CASCADA"][0] is not None) else edad_elegida
            mod_obj, ruta_m = REPO_MODELOS[cod_cem][clave_m]
            sigma_fb = cfg["incertidumbre_base"][edad_elegida]

            if mod_obj is not None:
                cols_req = features_activas + (['1_d', '3_d'] if clave_m == "28d_CASCADA" else [])
                y_p, p10, p90 = predecir_con_modelo(mod_obj, df_input_single, cols_req, sigma_fallback=sigma_fb)
                val_c, val_p10, val_p90 = float(y_p[0]), float(p10[0]), float(p90[0])

                col_res, col_shap = st.columns([1.0, 1.4])
                with col_res:
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:0.8rem; color:#9ca3af; text-transform:uppercase; margin-bottom:4px;">
                            {tipo_cemento_label} &bull; {edad_elegida.upper()}
                        </div>
                        <div class="result-value">{int(round(val_c))} <span class="result-unit">kg/cm²</span></div>
                        <div style="font-size:0.9rem; color:#d1d5db; margin-top:4px;">Valor exacto: <b>{val_c:.2f} kg/cm²</b></div>
                        <div class="conformal-badge">
                            🛡️ Banda Conformal 80%: <b>{val_p10:.1f}</b> a <b>{val_p90:.1f} kg/cm²</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_shap:
                    val_base, impactos = calcular_contribucion_local_kg(mod_obj, df_input_single, defaults, features_activas)
                    df_imp = pd.DataFrame({'Variable': list(impactos.keys()), 'Impacto_kg': list(impactos.values())})
                    df_imp['Abs'] = df_imp['Impacto_kg'].abs()
                    df_imp = df_imp.sort_values('Abs', ascending=True).tail(8)

                    colores = ['#10b981' if v >= 0 else '#E30613' for v in df_imp['Impacto_kg']]
                    fig_wf = go.Figure(go.Bar(
                        x=df_imp['Impacto_kg'], y=df_imp['Variable'],
                        orientation='h', marker_color=colores,
                        text=[f"{v:+.2f}" for v in df_imp['Impacto_kg']],
                        textposition='auto'
                    ))
                    fig_wf.update_layout(
                        title=f"Aporte Quimiométrico Local (vs Base: {val_base:.1f} kg/cm²)",
                        template="plotly_dark", height=280,
                        margin=dict(l=10, r=10, t=35, b=10),
                        xaxis_title="kg/cm²"
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.error(f"No se pudo cargar el archivo `.pkl` para **{cod_cem} - {edad_elegida}**.")
        else:
            edades_seq = ["1d", "3d", "7d", "28d"]
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
                    cols_cards[idx].markdown(f"""
                    <div class="result-card" style="padding:12px;">
                        <div style="font-size:0.75rem; color:#9ca3af;">EDAD {ed.upper()}</div>
                        <div class="result-value" style="font-size:2.2rem;">{int(round(yp[0]))} <span class="result-unit" style="font-size:0.9rem;">kg/cm²</span></div>
                        <div style="font-size:0.72rem; color:#9ca3af; margin-top:4px;">[{l10[0]:.0f} – {l90[0]:.0f}]</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    preds_c.append(np.nan)
                    preds_p10.append(np.nan)
                    preds_p90.append(np.nan)
                    cols_cards[idx].warning(f"Sin modelo {ed}")

            if not all(np.isnan(preds_c)):
                fig_curva = go.Figure()
                fig_curva.add_trace(go.Scatter(
                    x=dias_num + dias_num[::-1],
                    y=preds_p90 + preds_p10[::-1],
                    fill='toself', fillcolor='rgba(227, 6, 19, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Banda Conformal P10–P90'
                ))
                fig_curva.add_trace(go.Scatter(
                    x=dias_num, y=preds_c, mode='lines+markers+text',
                    line=dict(color='#E30613', width=3),
                    marker=dict(size=9, color='#ffffff', line=dict(width=2, color='#E30613')),
                    text=[f"{v:.0f}" if not np.isnan(v) else "" for v in preds_c],
                    textposition="top center", name='Curva Cinética'
                ))
                fig_curva.update_layout(
                    title=f"Curva de Desarrollo de Resistencia ({tipo_cemento_label})",
                    xaxis_title="Edad (Días)", yaxis_title="Resistencia (kg/cm²)",
                    template="plotly_dark", height=320
                )
                st.plotly_chart(fig_curva, use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 2: CARGA MASIVA SINCRONIZADA CON EL HORIZONTE
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown("##### 📂 Validación Masiva y Pronóstico de Lotes en Curado (`.xlsx`)")
    
    # Sincronización automática de la edad a evaluar según el horizonte superior seleccionado
    mapa_horizonte_batch = {
        "1 Día (Resistencia Temprana)": "1d",
        "3 Días (Control Operativo)": "3d",
        "7 Días (Desempeño Intermedio)": "7d",
        "28 Días (Certificación Final)": "28d"
    }
    edad_batch = mapa_horizonte_batch.get(horizonte, "3d")
    st.info(f"ℹ️ Edad de evaluación sincronizada automáticamente con el horizonte superior: **{edad_batch.upper()}**")

    archivo_subido = st.file_uploader(f"Sube tu Excel de {tipo_cemento_label}", type=["xlsx"])

    if archivo_subido is not None:
        df_lotes = pd.read_excel(archivo_subido)
        mod_b, _ = REPO_MODELOS[cod_cem][edad_batch]
        target_col = f"{edad_batch[:-1]}_d"

        if mod_b is not None:
            y_b, p10_b, p90_b = predecir_con_modelo(
                mod_b, df_lotes, cfg["features"], sigma_fallback=cfg["incertidumbre_base"][edad_batch]
            )
            df_out = df_lotes.copy()
            df_out.insert(0, f"{edad_batch}_PREDICTION", np.round(y_b, 2))
            df_out["P10_Min_Riesgo"] = np.round(p10_b, 2)
            df_out["P90_Max_Potencial"] = np.round(p90_b, 2)

            # Limpiar filas de promedios en la salida también
            if len(df_out) > 0:
                col_p = str(df_out.columns[0])
                df_out = df_out[~df_out[col_p].astype(str).str.lower().str.contains('promedio|average|media|total|sum', na=False)]

            if target_col in df_out.columns:
                y_real_b = pd.to_numeric(df_out[target_col], errors='coerce')
                mask_b = y_real_b.notna()

                df_out["MAE"] = np.where(mask_b, np.round(np.abs(y_real_b - y_b[:len(df_out)]), 2), np.nan)
                df_out["MAPE (%)"] = np.where(mask_b, np.round((df_out["MAE"] / y_real_b) * 100, 2), np.nan)

                if mask_b.sum() > 0:
                    mae_g = mean_absolute_error(y_real_b[mask_b], y_b[:len(df_out)][mask_b])
                    mape_g = np.mean(np.abs((y_real_b[mask_b] - y_b[:len(df_out)][mask_b]) / y_real_b[mask_b])) * 100
                    rmse_g = np.sqrt(mean_squared_error(y_real_b[mask_b], y_b[:len(df_out)][mask_b]))
                    st.session_state["ultimo_df_validado"] = (df_out, y_real_b[mask_b], y_b[:len(df_out)][mask_b], edad_batch)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE Promedio", f"{mae_g:.2f} kg/cm²")
                    m2.metric("MAPE Promedio", f"{mape_g:.2f}%")
                    m3.metric("RMSE Global", f"{rmse_g:.2f} kg/cm²")
                    m4.metric("Lotes Evaluados", f"{mask_b.sum()} reales / {(~mask_b).sum()} pronóstico")

            st.dataframe(df_out, use_container_width=True, height=340)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_out.to_excel(writer, index=False, sheet_name=f"Resultados_{cod_cem}_{edad_batch}")
            st.download_button(
                label=f"⬇️ Descargar Reporte Excel ({cod_cem} - {edad_batch})",
                data=buffer.getvalue(),
                file_name=f"DCCAI_MARCO_Resultados_{cod_cem}_{edad_batch}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.error(f"No se pudo cargar el archivo `.pkl` de **{cod_cem} - {edad_batch}**.")

# ------------------------------------------------------------------------------
# PESTAÑA 3: DIAGNÓSTICO SPC
# ------------------------------------------------------------------------------
with tab_spc:
    st.markdown("##### 📈 Auditoría de Residuales y Control Estadístico de Proceso (ASTM C109)")
    if "ultimo_df_validado" in st.session_state:
        df_v, y_r, y_p, ed_v = st.session_state["ultimo_df_validado"]
        residuales = y_p - y_r
        pendiente = np.polyfit(y_r, residuales, 1)[0]

        g1, g2 = st.columns(2)
        with g1:
            fig_sc = px.scatter(
                x=y_r, y=y_p, labels={'x': 'Real (kg/cm²)', 'y': 'Predicción DCCAI (kg/cm²)'},
                title=f"Predicción vs Real ({cod_cem} - {ed_v})"
            )
            min_v, max_v = min(y_r.min(), y_p.min()), max(y_r.max(), y_p.max())
            fig_sc.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="#E30613", dash="dash"))
            fig_sc.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_sc, use_container_width=True)

        with g2:
            fig_res = px.scatter(
                x=y_r, y=residuales, labels={'x': 'Real (Actual)', 'y': 'Residual (Pred - Real)'},
                title=f"Residuales vs Real (Pendiente = {pendiente:.3f})"
            )
            fig_res.add_hline(y=0, line_color="white", line_dash="solid")
            fig_res.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Sube un archivo Excel en la pestaña de Carga Masiva para habilitar los gráficos SPC.")

# Pie de página institucional limpio (sin repeticiones)
st.markdown("""
<div class="footer-disclaimer">
    <b>UNACEM &bull; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</b> &nbsp;|&nbsp; 
    DCCAI / QuAI — M.A.R.C.O. v2.0 &nbsp;|&nbsp; 
    Elaborado por: Ing. Marco Antonio Uriel Martínez Sánchez (Martínez Sánchez, Marco Antonio Uriel)<br>
    Las estimaciones e intervalos conformales son herramientas de soporte operativo y no sustituyen los ensayos físicos oficiales bajo norma ASTM C109.
</div>
""", unsafe_allow_html=True)
