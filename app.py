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
# 0. PARCHE DE COMPATIBILIDAD SKLEARN (COLAB <-> STREAMLIT CLOUD PYTHON 3.14)
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
    """Repara atributos internos (_fill_dtype / _fit_dtype) de SimpleImputer entre versiones de sklearn."""
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
    if hasattr(obj, "transformer_list"):
        for _, tr in obj.transformer_list:
            reparar_compatibilidad_sklearn(tr)
    if hasattr(obj, "model_"):
        reparar_compatibilidad_sklearn(obj.model_)
    if hasattr(obj, "base_estimator"):
        reparar_compatibilidad_sklearn(obj.base_estimator)
    return obj


# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y UTILIDAD DE LOGO EN BASE64
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


# ==============================================================================
# 2. ESTILOS CSS AVANZADOS + PANTALLA DE PRESENTACIÓN (SPLASH SCREEN)
# ==============================================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 96% !important;
    }
    .splash-box {
        background: radial-gradient(circle at center, #1f242e 0%, #0d0f14 100%);
        border: 1px solid #303642;
        border-top: 4px solid #E30613;
        border-radius: 16px;
        padding: 55px 30px;
        text-align: center;
        margin: 40px auto;
        max-width: 780px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.65);
    }
    .splash-dcca {
        font-size: 1.15rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    .splash-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
    }
    .splash-author {
        margin-top: 24px;
        padding-top: 18px;
        border-top: 1px solid #2d3340;
        font-size: 1.02rem;
        color: #e5e7eb;
    }
    .header-banner {
        background: linear-gradient(90deg, #14171f 0%, #1c212b 100%);
        border: 1px solid #2d3340;
        border-left: 5px solid #E30613;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .brand-tag {
        font-size: 0.85rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .main-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .author-badge {
        background: rgba(227, 6, 19, 0.12);
        border: 1px solid rgba(227, 6, 19, 0.4);
        border-radius: 8px;
        padding: 8px 14px;
        text-align: right;
        font-size: 0.82rem;
        color: #f3f4f6;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #d1d5db !important;
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
    }
    .result-card {
        background: radial-gradient(circle at top, #1f242e 0%, #11141a 100%);
        border: 1px solid #303642;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .result-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #f9fafb;
        line-height: 1;
    }
    .result-unit {
        font-size: 1.35rem;
        color: #E30613;
        font-weight: 700;
    }
    .conformal-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(227, 6, 19, 0.14);
        border: 1px solid rgba(227, 6, 19, 0.4);
        color: #f3f4f6;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .kpi-mini {
        background: #161922;
        border-left: 3px solid #E30613;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .kpi-mini-title {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .kpi-mini-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
    }
    .footer-disclaimer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.80rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1f242e;
    }
</style>
""", unsafe_allow_html=True)

# PANTALLA DE PRESENTACIÓN INICIAL (SPLASH SCREEN DE 3 SEGUNDOS)
if "splash_mostrado" not in st.session_state:
    placeholder = st.empty()
    logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="max-height:85px; margin-bottom:10px;">' if LOGO_B64 else '<div style="font-size:3rem;">🏭</div>'
    with placeholder.container():
        st.markdown(f"""
        <div class="splash-box">
            {logo_html}
            <div class="splash-dcca">UNACEM &bull; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>
            <div class="splash-title">DCCAI | QuAI &mdash; M.A.R.C.O. v2.0</div>
            <div style="color:#9ca3af; font-size:1.0rem;">
                Sistema Inteligente de Predicción de Resistencia a la Compresión y Optimización Quimiométrica (GU &bull; Tipo I &bull; HS)
            </div>
            <div class="splash-author">
                <b>Elaborado por:</b> Ing. Marco Antonio Uriel Martínez Sánchez<br>
                <span style="font-size:0.86rem; color:#9ca3af;">Martínez Sánchez, Marco Antonio Uriel</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        barra = st.progress(0, text="Inicializando motores quimiométricos de UNACEM - DCCA...")
        for pct in range(0, 101, 25):
            time.sleep(0.40)
            barra.progress(pct, text="Cargando modelos predictivos (Cemento Tipo GU, Tipo I y Tipo HS)...")
    placeholder.empty()
    st.session_state["splash_mostrado"] = True


# ==============================================================================
# 3. CATÁLOGO QUIMIOMÉTRICO POR TIPO DE CEMENTO (GU: 16, TI: 15, HS: 18)
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


# ==============================================================================
# 4. CARGA ROBUSTA DE LOS MODELOS (.PKL) CON REPARACIÓN AUTOMÁTICA
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
# 5. MOTOR DE INFERENCIA A PRUEBA DE FALLOS DE VERSIÓN SKLEARN
# ==============================================================================
def normalizar_nombre(texto):
    return re.sub(
        r'[^a-z0-9]', '',
        unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()
    )

def alinear_dataframe(modelo, df_input, features_esperadas):
    expected_cols = None
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
    if not expected_cols:
        expected_cols = features_esperadas

    mapa_limpio = {normalizar_nombre(c): c for c in df_input.columns}
    df_alineado = pd.DataFrame(index=df_input.index)

    for col in expected_cols:
        col_norm = normalizar_nombre(col)
        if col in df_input.columns:
            df_alineado[col] = pd.to_numeric(df_input[col], errors='coerce')
        elif col_norm in mapa_limpio:
            df_alineado[col] = pd.to_numeric(df_input[mapa_limpio[col_norm]], errors='coerce')
        else:
            df_alineado[col] = 0.0
    return df_alineado

def ejecutar_predict_seguro(modelo, df_aln):
    """Ejecuta predict reparando SimpleImputer o saltando al estimador final si cambia la API de sklearn."""
    reparar_compatibilidad_sklearn(modelo)
    try:
        return np.asarray(modelo.predict(df_aln), dtype=float)
    except AttributeError:
        # Respaldo directo: imputa -9999 manualmente y ejecuta los pasos restantes del Pipeline
        X_val = df_aln.fillna(-9999.0).values
        if hasattr(modelo, "steps"):
            for nombre_paso, paso in modelo.steps[:-1]:
                if not isinstance(paso, SimpleImputer):
                    X_val = paso.transform(X_val)
            return np.asarray(modelo.steps[-1][1].predict(X_val), dtype=float)
        raise

def predecir_con_modelo(modelo, df_input, features_esperadas, sigma_fallback=10.0):
    df_aln = alinear_dataframe(modelo, df_input, features_esperadas)
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
# 6. BANNER INSTITUCIONAL PERMANENTE (UNACEM + DCCA + AUTOR)
# ==============================================================================
logo_header_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:54px; margin-right:16px;">' if LOGO_B64 else ''

st.markdown(f"""
<div class="header-banner">
    <div style="display:flex; align-items:center;">
        {logo_header_html}
        <div>
            <div class="brand-tag">UNACEM &bull; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</div>
            <div class="main-title">DCCAI | QuAI &mdash; M.A.R.C.O. v2.0</div>
            <div style="font-size:0.84rem; color:#9ca3af;">
                Predicción Multi-Edad de Resistencia a la Compresión (1, 3, 7 y 28 Días) y Auditoría Quimiométrica XRD
            </div>
        </div>
    </div>
    <div class="author-badge">
        <div style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Elaborado por</div>
        <div style="font-weight:700; color:#ffffff; font-size:0.92rem;">Ing. Marco Antonio Uriel Martínez Sánchez</div>
        <div style="font-size:0.76rem; color:#d1d5db;">Martínez Sánchez, Marco Antonio Uriel</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sel1, col_sel2, col_sel3 = st.columns([1.4, 1.4, 1.2])
with col_sel1:
    tipo_cemento_label = st.selectbox(
        "Familia de Cemento",
        list(CONFIG_CEMENTOS.keys()),
        index=2
    )
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
    usar_cascada_28d = st.toggle(
        "Activar Modo Cascada 28d (Inyectar 1d/3d real)",
        value=False
    )


# ==============================================================================
# 7. PESTAÑAS DE OPERACIÓN INDUSTRIAL
# ==============================================================================
tab_sim, tab_batch, tab_spc = st.tabs([
    "🎛️ Simulador Quimiométrico & Explicabilidad SHAP",
    "📂 Carga Masiva Excel & Evaluación Ciega (MAE / MAPE / RMSE)",
    "📈 Diagnóstico de Residuales & Control Estadístico (ASTM C109)"
])

with tab_sim:
    defaults = cfg["defaults"]
    features_activas = cfg["features"]

    st.markdown("##### 🧪 Matriz de Composición Fisicoquímica y Fases XRD Rietveld")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        caliza = st.number_input("%Caliza", value=float(defaults['%Caliza']), step=0.10, format="%.2f")
        so3 = st.number_input("SO3 (%)", value=float(defaults['SO3']), step=0.05, format="%.2f")
        cal_libre = st.number_input("Cal Libre", value=float(defaults['Cal_Libre']), step=0.05, format="%.2f")
        hemidrato = st.number_input("Hemidrato", value=float(defaults['Hemidrato']), step=0.10, format="%.2f")
        if cod_cem == "HS":
            puzolana = st.number_input("%Puzolana", value=float(defaults['%Puzolana']), step=0.20, format="%.2f")
        else:
            puzolana = 0.0

    with c2:
        yeso = st.number_input("%Yeso", value=float(defaults['%Yeso']), step=0.10, format="%.2f")
        alcali_eq = st.number_input("Álcali_Eq", value=float(defaults['Álcali_Eq']), step=0.01, format="%.2f")
        c3a_cub = st.number_input("C3A Cúbico", value=float(defaults['C3A_Cúbico']), step=0.10, format="%.2f")
        anhidrita = st.number_input("Anhidrita", value=float(defaults['Anhidrita']), step=0.05, format="%.2f")
        if cod_cem == "HS":
            escoria = st.number_input("%Escoria", value=float(defaults['%Escoria']), step=0.20, format="%.2f")
        else:
            escoria = 0.0

    with c3:
        m325 = st.number_input("M325 (%)", value=float(defaults['M325']), step=0.05, format="%.2f")
        alita = st.number_input("Alita (C3S)", value=float(defaults['Alita']), step=0.50, format="%.2f")
        c3a_orto = st.number_input("C3A Ortorrómbico", value=float(defaults['C3A_Ortorómbico']), step=0.10, format="%.2f")
        if cod_cem in ["GU", "HS"]:
            agua_cemento = st.number_input("Agua/Cemento (%)", value=float(defaults['Agua_Cemento']), step=0.20, format="%.2f")
        else:
            agua_cemento = 48.50
            st.caption("ℹ️ *En Tipo I la relación A/C es fija por norma (0.485).*")

    with c4:
        blaine = st.number_input("Blaine (cm²/g)", value=float(defaults['Blaine']), step=10.0, format="%.1f")
        belita = st.number_input("Belita (C2S)", value=float(defaults['Belita']), step=0.50, format="%.2f")
        gypsum = st.number_input("Gypsum", value=float(defaults['Gypsum']), step=0.10, format="%.2f")
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

    alertas_rango = []
    for var_chk, (vmin, vmax) in cfg["rangos_3sigma"].items():
        val_u = registro_actual.get(var_chk, 0.0)
        if val_u < vmin or val_u > vmax:
            alertas_rango.append(f"**{var_chk}** ({val_u}) fuera de banda habitual [{vmin} – {vmax}]")
    if alertas_rango:
        st.warning("⚠️ **Alerta de Dominio Quimiométrico:** " + " | ".join(alertas_rango))

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

                col_res, col_shap = st.columns([1.05, 1.45])
                with col_res:
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:0.85rem; color:#9ca3af; text-transform:uppercase; margin-bottom:6px;">
                            {tipo_cemento_label} &bull; {edad_elegida.upper()} ({ruta_m})
                        </div>
                        <div class="result-value">{int(round(val_c))} <span class="result-unit">kg/cm²</span></div>
                        <div style="font-size:0.95rem; color:#d1d5db; margin-top:6px;">Valor exacto: <b>{val_c:.2f} kg/cm²</b></div>
                        <div class="conformal-badge">
                            🛡️ Banda Conformal 80% (P10 – P90): <b>{val_p10:.1f}</b> a <b>{val_p90:.1f} kg/cm²</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_shap:
                    val_base, impactos = calcular_contribucion_local_kg(mod_obj, df_input_single, defaults, features_activas)
                    df_imp = pd.DataFrame({
                        'Variable': list(impactos.keys()),
                        'Impacto_kg': list(impactos.values())
                    })
                    df_imp['Abs'] = df_imp['Impacto_kg'].abs()
                    df_imp = df_imp.sort_values('Abs', ascending=True).tail(10)

                    colores = ['#10b981' if v >= 0 else '#E30613' for v in df_imp['Impacto_kg']]
                    fig_wf = go.Figure(go.Bar(
                        x=df_imp['Impacto_kg'], y=df_imp['Variable'],
                        orientation='h', marker_color=colores,
                        text=[f"{v:+.2f} kg/cm²" for v in df_imp['Impacto_kg']],
                        textposition='auto'
                    ))
                    fig_wf.update_layout(
                        title=f"Explicabilidad Local (Aporte en kg/cm² vs Estándar: {val_base:.1f} kg/cm²)",
                        template="plotly_dark", height=310,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_title="Impacto Neto (kg/cm²)"
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
                    <div class="result-card" style="padding:16px;">
                        <div style="font-size:0.8rem; color:#9ca3af;">EDAD {ed.upper()}</div>
                        <div class="result-value" style="font-size:2.4rem;">{int(round(yp[0]))} <span class="result-unit" style="font-size:1rem;">kg/cm²</span></div>
                        <div style="font-size:0.78rem; color:#9ca3af; margin-top:6px;">P10–P90: [{l10[0]:.0f} – {l90[0]:.0f}]</div>
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
                    marker=dict(size=10, color='#ffffff', line=dict(width=2, color='#E30613')),
                    text=[f"{v:.0f}" if not np.isnan(v) else "" for v in preds_c],
                    textposition="top center", name='Curva Cinética Predicha'
                ))
                fig_curva.update_layout(
                    title=f"Curva de Desarrollo de Resistencia ({tipo_cemento_label})",
                    xaxis_title="Edad de Curado (Días)", yaxis_title="Resistencia a Compresión (kg/cm²)",
                    template="plotly_dark", height=360
                )
                st.plotly_chart(fig_curva, use_container_width=True)

with tab_batch:
    st.markdown("##### 📂 Validación Masiva y Pronóstico de Lotes en Curado (`.xlsx`)")
    archivo_subido = st.file_uploader(
        f"Sube tu Excel de {tipo_cemento_label} (las filas sin resistencia real se pronostican sin alterar el MAE/MAPE)",
        type=["xlsx"]
    )
    edad_batch = st.radio("Selecciona la edad a evaluar en el Excel:", ["1d", "3d", "7d", "28d"], horizontal=True)

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

            if target_col in df_out.columns:
                y_real_b = pd.to_numeric(df_out[target_col], errors='coerce')
                mask_b = y_real_b.notna()

                df_out["MAE"] = np.where(mask_b, np.round(np.abs(y_real_b - y_b), 2), np.nan)
                df_out["MAPE (%)"] = np.where(mask_b, np.round((df_out["MAE"] / y_real_b) * 100, 2), np.nan)
                df_out["RMSE_Fila"] = np.where(mask_b, np.round((y_real_b - y_b) ** 2, 2), np.nan)

                if mask_b.sum() > 0:
                    mae_g = mean_absolute_error(y_real_b[mask_b], y_b[mask_b])
                    mape_g = np.mean(np.abs((y_real_b[mask_b] - y_b[mask_b]) / y_real_b[mask_b])) * 100
                    rmse_g = np.sqrt(mean_squared_error(y_real_b[mask_b], y_b[mask_b]))
                    st.session_state["ultimo_df_validado"] = (df_out, y_real_b[mask_b], y_b[mask_b], edad_batch)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE Promedio", f"{mae_g:.2f} kg/cm²")
                    m2.metric("MAPE Promedio", f"{mape_g:.2f}%")
                    m3.metric("RMSE Global", f"{rmse_g:.2f} kg/cm²")
                    m4.metric("Lotes Evaluados / En Curado", f"{mask_b.sum()} reales / {(~mask_b).sum()} pronóstico")

            st.dataframe(df_out, use_container_width=True, height=360)

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

with tab_spc:
    st.markdown("##### 📈 Auditoría de Residuales y Control Estadístico de Proceso (SPC)")
    if "ultimo_df_validado" in st.session_state:
        df_v, y_r, y_p, ed_v = st.session_state["ultimo_df_validado"]
        residuales = y_p - y_r
        pendiente = np.polyfit(y_r, residuales, 1)[0]

        g1, g2 = st.columns(2)
        with g1:
            fig_sc = px.scatter(
                x=y_r, y=y_p, labels={'x': 'Resistencia Real Prensa (kg/cm²)', 'y': 'Predicción DCCAI (kg/cm²)'},
                title=f"Predicción vs Real ({cod_cem} - {ed_v})"
            )
            min_v, max_v = min(y_r.min(), y_p.min()), max(y_r.max(), y_p.max())
            fig_sc.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="#E30613", dash="dash"))
            fig_sc.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_sc, use_container_width=True)

        with g2:
            fig_res = px.scatter(
                x=y_r, y=residuales,
                labels={'x': 'Resistencia Real (Actual)', 'y': 'Residual (Predicho - Real)'},
                title=f"Residuales vs Real (Pendiente = {pendiente:.3f})"
            )
            fig_res.add_hline(y=0, line_color="white", line_dash="solid")
            fig_res.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Sube un archivo Excel en la pestaña **Carga Masiva Excel** para visualizar la ortogonalidad de residuales y las cartas de control.")

# Pie de página institucional
st.markdown("""
<div class="footer-disclaimer">
    <b>UNACEM &bull; DIVISIÓN DE CONTROL DE CALIDAD ATOCONGO - DCCA</b> &nbsp;|&nbsp;
    <b>DCCAI / QuAI presenta M.A.R.C.O. v2.0</b> (<i>Machine learning Algorithm for Resistance and Cement Optimization</i>)<br>
    <b>Elaborado por: Ing. Marco Antonio Uriel Martínez Sánchez</b> (Martínez Sánchez, Marco Antonio Uriel)<br>
    Las estimaciones e intervalos conformales son herramientas de soporte operativo y no sustituyen los ensayos físicos oficiales bajo norma ASTM C109.
</div>
""", unsafe_allow_html=True)
