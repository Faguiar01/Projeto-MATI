import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="MATI Dashboard", layout="wide")

st.title("🚀 MATI - Real Time Analytics")
st.write("Monitoramento de Eye Aspect Ratio (EAR)")

placeholder = st.empty()

while True:
    try:
        df = pd.read_csv("dados_mati.csv")
        
        with placeholder.container():
            col1, col2 = st.columns(2)
            
            # Métrica atual
            ear_atual = df['ear'].iloc[-1]
            status = df['status'].iloc[-1]
            
            col1.metric("EAR Atual", ear_atual)
            col2.metric("Status", status)

            # Gráfico de linha (últimos 100 registros)
            st.subheader("Gráfico de Atenção (EAR)")
            st.line_chart(df['ear'].tail(100))
            
    except Exception as e:
        st.error(f"Aguardando dados... {e}")
    
    time.sleep(0.5)