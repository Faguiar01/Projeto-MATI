# ============================================================
# PROJETO MATI — dashboard_mati.py
# Versão 2.0 — Monitoramento de Fadiga e Emoções
# ============================================================

import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Dashboard MATI", page_icon="📊", layout="wide")

st.title("📊 Projeto MATI - Dashboard 5.0")
st.write("Painel de monitoramento da interação humana: Fadiga e Emoções em tempo real.")

ARQUIVO_CSV = "dados_mati.csv"

def carregar_dados():
    try:
        df = pd.read_csv(ARQUIVO_CSV)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# Tenta carregar os dados
df = carregar_dados()

# Botão para atualizar a tela
if st.button("🔄 Atualizar Dados"):
    st.rerun()

if not df.empty:
    st.divider()
    
    # --- BLOCO 1: INDICADORES (KPIs) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total de Leituras", value=len(df))
    
    with col2:
        alertas = len(df[df['status_fadiga'] == 'Fadiga'])
        st.metric(label="Alertas de Fadiga (Micro-sonos)", value=alertas)
        
    with col3:
        if 'emocao' in df.columns:
            # Calcula a emoção que mais apareceu
            emocao_predominante = df['emocao'].mode()[0] if not df['emocao'].empty else "N/A"
            st.metric(label="Humor Predominante", value=emocao_predominante)

    st.divider()

    # --- BLOCO 2: GRÁFICOS VISUAIS ---
    col_grafico1, col_grafico2 = st.columns(2)

    with col_grafico1:
        st.subheader("👁️ Nível de Atenção (EAR)")
        st.write("Gráfico de linha mostrando a abertura dos olhos.")
        # Usamos o horário como índice para o gráfico de linha
        df_ear = df.set_index('horario')
        st.line_chart(df_ear['ear'])

    with col_grafico2:
        if 'emocao' in df.columns:
            st.subheader("🧠 Distribuição de Emoções")
            st.write("Quais foram as emoções mais detectadas?")
            # Conta quantas vezes cada emoção apareceu
            contagem_emocoes = df['emocao'].value_counts()
            st.bar_chart(contagem_emocoes)
            
    st.divider()
    
    # --- BLOCO 3: BANCO DE DADOS BRUTO ---
    st.subheader("📋 Tabela de Registros (Últimas 15 leituras)")
    st.dataframe(df.tail(15), width='stretch')

else:
    st.warning("⚠️ Nenhum dado encontrado. Rode o arquivo 'camera_mati.py' primeiro para gerar o arquivo CSV.")