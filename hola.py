import streamlit as st
import pandas as pd
import numpy as np


st.title("📊 Dashboards Inteligentes")

np.random.seed(9)
df = pd.DataFrame({
    "fecha": pd.date_range("2024-01-01", periods=60),
    "ingresos": np.random.normal(10000, 1500, 60),
    "gastos": np.random.normal(6000, 800, 60)
})

df["utilidad"] = df["ingresos"] - df["gastos"]

fig = st.line_chart(df, x="fecha", y=["ingresos", "gastos", "utilidad"],
              title="Dashboard financiero")
st.plotly_chart(fig, use_container_width=True)
