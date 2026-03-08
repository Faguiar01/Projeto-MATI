import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import subprocess
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="MATI Dashboard Executivo", layout="wide")

# --- CABEÇALHO COM LOGO E TÍTULO ---
col_logo, col_titulo = st.columns([1, 4])

with col_logo:
    if os.path.exists("logo_mati.png"):
        st.image("logo_mati.png", width=150)
    else:
        st.markdown("### [ MATI ]")

with col_titulo:
    st.title("Monitoramento de Atenção, Tensão e Interação")
    st.markdown("### Gestão Humanizada & Indústria 5.0")

st.markdown("---")

ARQUIVO_CSV = 'dados_mati.csv'

# --- BARRA LATERAL (CONTROLE GERAL) ---
st.sidebar.header("Central de Comando")
if st.sidebar.button("📄 GERAR RELATÓRIO PDF (LAUDO)"):
    try:
        # Chama o script de relatório que você já tem calibrado
        subprocess.run(["python", "gerar_relatorio.py"], check=True)
        st.sidebar.success("✅ Laudo_MATI.pdf gerado com sucesso!")
        
        # Opcional: Abrir o PDF automaticamente (Windows)
        os.startfile("Laudo_MATI.pdf")
    except Exception as e:
        st.sidebar.error(f"Erro ao gerar PDF: {e}")

st.sidebar.markdown("---")
st.sidebar.info("O sistema está capturando dados do sensor rPPG e EAR em tempo real.")

# --- CORPO DO DASHBOARD ---
placeholder = st.empty()

while True:
    if os.path.exists(ARQUIVO_CSV):
        try:
            df = pd.read_csv(ARQUIVO_CSV)
            if not df.empty:
                # Dados atuais
                operador = str(df['colaborador'].iloc[-1]).upper()
                ultimo_ear = df.iloc[-1]['ear']
                ultimo_bpm = df.iloc[-1]['bpm']
                ultima_emocao = df.iloc[-1]['emocao']
                status_fadiga = df.iloc[-1]['status_fadiga']
                
                df_recente = df.tail(60)

                with placeholder.container():
                    st.subheader(f"👤 Operador: {operador}")
                    
                    # KPIs
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Atenção (EAR)", f"{ultimo_ear:.3f}")
                    kpi2.metric("Frequência (BPM)", f"{int(ultimo_bpm)} BPM")
                    kpi3.metric("Estado", ultima_emocao)
                    
                    cor_status = "🔴 FADIGA" if status_fadiga == "Fadiga" else "🟢 NORMAL"
                    kpi4.metric("Status Segurança", cor_status)

                    # Gráficos
                    g1, g2 = st.columns(2)
                    
                    with g1:
                        fig_ear = go.Figure()
                        fig_ear.add_trace(go.Scatter(x=df_recente['horario'], y=df_recente['ear'], name="EAR", line=dict(color='cyan')))
                        fig_ear.update_layout(title="Nível de Atenção Visual", template="plotly_dark", height=300)
                        st.plotly_chart(fig_ear, use_container_width=True)
                        
                    with g2:
                        fig_rppg = go.Figure()
                        fig_rppg.add_trace(go.Scatter(x=df_recente['horario'], y=df_recente['sinal_verde'], name="Sinal rPPG", line=dict(color='lime')))
                        fig_rppg.update_layout(title="Onda de Pulso (Sensor Óptico)", template="plotly_dark", height=300)
                        st.plotly_chart(fig_rppg, use_container_width=True)
            
            time.sleep(1)
            
        except:
            pass
    else:
        st.error("Aguardando arquivo de dados...")
        time.sleep(2)