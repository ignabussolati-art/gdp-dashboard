import streamlit as st

# ---- CONFIGURACIÓN ----
st.set_page_config(
    page_title="Consultoría Financiera con IA",
    page_icon="📊",
    layout="wide"
)

# ---- ESTILOS CSS (para que parezca una landing real) ----
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: white;
}
.section {
    background: #1e293b;
    padding: 40px;
    border-radius: 18px;
    margin-bottom: 40px;
    border: 1px solid #334155;
}
.code-block {
    background: #0f172a;
    padding: 16px;
    border-radius: 12px;
    overflow-x: auto;
    border: 1px solid #334155;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---- HERO SECTION ----
st.markdown("""
<div style="text-align:center; padding: 60px 0;">
    <h1 style="font-size:3rem; font-weight:800;">Consultoría Financiera con Inteligencia Artificial</h1>
    <p style="font-size:1.3rem; color:#cbd5e1;">
        Optimiza tus finanzas empresariales con análisis automatizados, predicciones avanzadas y dashboards inteligentes.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- SERVICIOS ----
st.header("Servicios impulsados por IA")
st.write("### A continuación, los servicios detallados con el código que los implementa:")

# Diccionario de servicios y código asociado
servicios = {
    "1. Análisis Financiero Automatizado": """
def analizar_finanzas(df):
    revenue = df[df['cuenta']=='ventas']['monto'].sum()
    cogs = df[df['cuenta']=='costo_ventas']['monto'].sum()
    gastos = df[df['cuenta']=='gastos_operativos']['monto'].sum()
    margen = (revenue - cogs) / revenue if revenue else 0
    return {
        'margen_bruto': margen,
        'ebitda': revenue - cogs - gastos
    }
""",

    "2. Modelos de Predicción Financiera": """
from sklearn.linear_model import LinearRegression

def predecir_flujo(df):
    df['lag_1'] = df['ventas'].shift(1)
    df = df.dropna()
    X = df[['lag_1']]
    y = df['ventas']
    model = LinearRegression().fit(X, y)
    pred = model.predict([[df['ventas'].iloc[-1]]])[0]
    return pred
""",

    "3. Optimización de Presupuestos": """
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

def optimizar_presupuesto(costos, requeridos, max_total):
    prob = LpProblem('BudgetOpt', LpMinimize)
    alloc = {a: LpVariable(a, lowBound=requeridos[a]) for a in costos}
    prob += lpSum([costos[a]*alloc[a] for a in costos])
    prob += lpSum([alloc[a] for a in costos]) <= max_total
    prob.solve()
    return {a: alloc[a].value() for a in costos}
""",

    "4. Gestión de Riesgos con IA": """
from sklearn.ensemble import GradientBoostingClassifier

def score_riesgo(df):
    X = df[['edad','ingresos','deuda_ratio']]
    y = df['default']
    model = GradientBoostingClassifier().fit(X, y)
    df['score'] = model.predict_proba(X)[:,1]
    return df[['cliente_id','score']]
""",

    "5. Automatización Financiera": """
def conciliar(banco, libro):
    banco['key'] = banco['monto'].round(2)
    libro['key'] = libro['monto'].round(2)
    merged = banco.merge(libro, on='key')
    return merged
""",

    "6. Asesoría Estratégica con IA": """
import numpy as np

def simular_escenarios(base, meses=12):
    proy = []
    cash = base
    for m in range(meses):
        ingreso = np.random.normal(10000,2000)
        gasto = np.random.normal(8000,1500)
        cash += ingreso - gasto
        proy.append(cash)
    return proy
""",

    "7. Dashboards Financieros Inteligentes": """
import plotly.express as px

def generar_dashboard(df):
    fig = px.line(df, x='fecha', y='valor', color='kpi')
    return fig
"""
}

# ---- Mostrar servicios en bloques ----
for titulo, codigo in servicios.items():
    st.markdown(f"<div class='section'><h3>{titulo}</h3>", unsafe_allow_html=True)
    st.markdown(f"<pre class='code-block'>{codigo}</pre></div>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
<div style="text-align:center; padding:40px; color:#94a3b8;">
    © 2025 Consultoría Financiera con IA — Todos los derechos reservados.
</div>
""", unsafe_allow_html=True)
