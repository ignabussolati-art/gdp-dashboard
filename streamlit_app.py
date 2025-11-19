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
