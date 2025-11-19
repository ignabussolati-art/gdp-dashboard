import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Consultoría Financiera con IA",
    page_icon="📊",
    layout="wide"
)

# ---- Estilos CSS ----
def load_css():
    st.markdown("""
    <style>
    body {
        background-color: #f6f8fa;
    }
    .hero {
        padding: 80px 20px;
        text-align: center;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        border-radius: 20px;
        margin-bottom: 40px;
    }
    .card {
        background: #f6f8fa;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    code {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ---- HERO ----
st.markdown("""
<div class="hero">
    <h1 style="font-size: 42px; font-weight: 900;">Consultoría Financiera impulsada por IA</h1>
    <p style="font-size: 22px; margin-top: 10px;">
        Optimiza tus decisiones financieras con análisis automatizado, modelos predictivos y dashboards inteligentes.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Beneficios ----
st.header("🔍 ¿Qué obtiene tu empresa?")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'> <h3>Análisis en segundos</h3> Identifica oportunidades y anomalías automáticamente. </div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'> <h3>Predicciones precisas</h3> Modelos ML para ventas, flujo de caja y demanda. </div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'> <h3>Automatización total</h3> Reportes, conciliaciones y dashboards inteligentes. </div>", unsafe_allow_html=True)

st.write("---")

# ---- Servicios con Código ----
st.header("🧠 Servicios con IA + Código Real en Python")

accordion = st.expander("1️⃣ Análisis Financiero Automatizado", expanded=False)
with accordion:
    st.code("""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

df = pd.read_csv('estados_financieros.csv', parse_dates=['fecha'])

revenue = df[df['cuenta']=='ventas']['monto'].sum()
cogs = df[df['cuenta']=='costo_ventas']['monto'].sum()
operating = df[df['cuenta']=='gastos_operativos']['monto'].sum()

kpis = {
    "margen_bruto": (revenue - cogs) / revenue,
    "ebitda": revenue - cogs - operating
}

# Detección de anomalías
trans = df.groupby(['fecha'])['monto'].sum().reset_index()
model = IsolationForest(contamination=0.01, random_state=42)
trans['anomaly'] = model.fit_predict(trans[['monto']])
""")

accordion = st.expander("2️⃣ Predicción Financiera (Ventas / Cashflow)", expanded=False)
with accordion:
    st.code("""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

sales = pd.read_csv('ventas.csv', parse_dates=['fecha']).set_index('fecha')

sales['lag_7'] = sales['ventas'].shift(7)
sales['ma_14'] = sales['ventas'].rolling(14).mean()
sales = sales.dropna()

X = sales[['lag_7','ma_14']]
y = sales['ventas']

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
prediction = model.predict(X.tail(1))[0]
""")

accordion = st.expander("3️⃣ Optimización de Presupuestos", expanded=False)
with accordion:
    st.code("""
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

areas = ['marketing','ventas','ops']
costs = {'marketing':1.0, 'ventas':1.2, 'ops':0.8}
min_req = {'marketing':1000, 'ventas':2000, 'ops':1500}
max_total = 6000

prob = LpProblem('Optimizar', LpMinimize)
alloc = {a: LpVariable(a, lowBound=min_req[a]) for a in areas}

prob += lpSum(costs[a]*alloc[a] for a in areas)
prob += lpSum(alloc[a] for a in areas) <= max_total
prob.solve()
""")

accordion = st.expander("4️⃣ Gestión de Riesgos (Credit Score)", expanded=False)
with accordion:
    st.code("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

df = pd.read_csv('clientes.csv')
X = df[['edad','ingresos','deuda_ratio']]
y = df['default']

model = GradientBoostingClassifier()
model.fit(X, y)
df['score'] = model.predict_proba(X)[:,1]
""")

accordion = st.expander("5️⃣ Automatización Financiera", expanded=False)
with accordion:
    st.code("""
bank = pd.read_csv('banco.csv')
ledger = pd.read_csv('libro.csv')

bank['key'] = bank['monto'].round(2).astype(str)
ledger['key'] = ledger['monto'].round(2).astype(str)

matched = bank.merge(ledger, on='key')
""")

accordion = st.expander("6️⃣ Asesoría Estratégica (Monte Carlo)", expanded=False)
with accordion:
    st.code("""
import numpy as np
base_cash = 100000
n_sim = 1000
horizon = 12

results = np.zeros((n_sim,horizon))
for i in range(n_sim):
    cash = base_cash
    for m in range(horizon):
        ingreso = np.random.normal(10000,2000)
        gasto = np.random.normal(8000,1500)
        cash += ingreso - gasto
        results[i,m] = cash
""")

accordion = st.expander("7️⃣ Dashboards Inteligentes", expanded=False)
with accordion:
    st.code("""
from dash import Dash, dcc, html
import plotly.express as px

# Para dashboards avanzados fuera de Streamlit
""")

st.write("---")

# ---- Formulario ----
st.header("📅 Agenda una Demo Gratis")

with st.form("Demo"):
    name = st.text_input("Nombre")
    email = st.text_input("Correo")
    company = st.text_input("Empresa")
    needs = st.text_area("¿Qué necesitas?")
    submitted = st.form_submit_button("Enviar solicitud")

    if submitted:
        st.success("¡Gracias! Nos contactaremos contigo muy pronto.")

# ---- Footer ----
st.write("---")
st.caption("© 2025 Consultoría Financiera con IA — Todos los derechos reservados")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("🔍 Análisis Financiero Automatizado")

# Datos ficticios
np.random.seed(0)
dates = pd.date_range("2024-01-01", periods=90)
gastos = np.random.normal(5000, 900, 90)
ingresos = np.random.normal(9000, 1500, 90)

df = pd.DataFrame({
    "fecha": dates,
    "ingresos": ingresos,
    "gastos": gastos,
    "utilidad": ingresos - gastos
})

# KPIs
st.subheader("📌 KPIs Financieros")
col1, col2, col3 = st.columns(3)
col1.metric("Ingresos totales", f"${df['ingresos'].sum():,.0f}")
col2.metric("Gastos totales", f"${df['gastos'].sum():,.0f}")
col3.metric("Utilidad neta", f"${df['utilidad'].sum():,.0f}")

# Gráfico utilidad
st.subheader("📈 Utilidad diaria")
fig = px.line(df, x="fecha", y="utilidad")
st.plotly_chart(fig, use_container_width=True)

# Detección de anomalías simple
threshold = df["utilidad"].mean() - 2 * df["utilidad"].std()
anomalias = df[df["utilidad"] < threshold]

st.subheader("⚠️ Anomalías detectadas")
st.dataframe(anomalias)
