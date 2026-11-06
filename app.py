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
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
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
    def __init__(self, base_estimator=None, gamma_stretch=0.0, alpha_conformal=0.20, cv=5):
        self.base_estimator = base_estimator
        self.gamma_stretch = gamma_stretch
        self.alpha_conformal = alpha_conformal
        self.cv = cv
    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float)
        self.feature_names_in_ = list(X_df.columns)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=1234)
        oof_preds = np.zeros_like(y_arr)
        for tr_idx, val_idx in kf.split(X_df):
            est_c = clone(self.base_estimator)
            est_c.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
            oof_preds[val_idx] = est_c.predict(X_df.iloc[val_idx])
        self.mu_y_ = float(np.mean(y_arr))
        self.std_y_ = float(np.std(y_arr) + 1e-6)
        self.mu_pred_ = float(np.mean(oof_preds))
        self.std_pred_ = float(np.std(oof_preds) + 1e-6)
        ratio_std = np.clip(self.std_y_ / self.std_pred_, 1.0, 2.2)
        self.factor_escala_ = float(1.0 + self.gamma_stretch * (ratio_std - 1.0))
        oof_cal = self.mu_y_ + self.factor_escala_ * (oof_preds - self.mu_pred_)
        res_abs = np.abs(y_arr - oof_cal)
        n = len(y_arr)
        q_lvl = np.ceil((n + 1) * (1.0 - self.alpha_conformal)) / n
        self.q_conformal_ = float(np.quantile(res_abs, min(q_lvl, 0.95)))
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_df, y_arr)
        return self
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
    if hasattr(obj, "transformer_list"):
        for _, tr in obj.transformer_list:
            reparar_compatibilidad_sklearn(tr)
    if hasattr(obj, "estimators_"):
        for est in obj.estimators_:
            reparar_compatibilidad_sklearn(est)
    if hasattr(obj, "model_"):
        reparar_compatibilidad_sklearn(obj.model_)
    if hasattr(obj, "base_estimator"):
        reparar_compatibilidad_sklearn(obj.base_estimator)
    return obj


# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS EJECUTIVOS
# ==============================================================================
st.set_page_config(
    page_title="DCCAI: Inteligencia Artificial para el Control de Calidad — UNACEM",
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
        padding-bottom: 5.0rem !important;
        max-width: 97% !important;
    }
    .splash-box {
        background: radial-gradient(circle at center, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-top: 4px solid #E30613;
        border-radius: 18px;
        padding: 48px 35px;
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
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 50%, #1E293B 100%);
        border: 1px solid #334155;
        border-top: 3px solid #E30613;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 16px;
        display: grid;
        grid-template-columns: 1fr 2.4fr 1fr;
        align-items: center;
        gap: 14px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.30);
    }
    @media (max-width: 992px) {
        .header-banner {
            grid-template-columns: 1fr;
            text-align: center;
        }
    }
    .header-left {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .header-center {
        text-align: center;
    }
    .header-right {
        display: flex;
        justify-content: flex-end;
    }
    .brand-tag {
        font-size: 0.76rem;
        font-weight: 800;
        color: #E30613;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .main-title {
        font-size: 1.56rem;
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
        padding: 9px 15px;
        text-align: center;
        font-weight: 700;
        color: #F8FAFC;
        font-size: 0.85rem;
        line-height: 1.3;
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
