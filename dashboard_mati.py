# ============================================================
# PROJETO MATI — dashboard.py
# Versão 6.1 — Painel de Gestão com Identificação de Operador
# ============================================================

import streamlit as st
import pandas as pd
import time

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Dashboard MATI v6.1", layout="wide")

# 2. FUNÇÃO DE LEITURA DE DADOS
ARQUIVO_CSV = "dados_mati.csv"

def carregar_dados():
    try:
        df = pd.read_csv(ARQUIVO_CSV)
        return df.tail(100)
    except Exception as e:
        st.error(f"Erro técnico ao tentar ler o CSV: {e}")
        return pd.DataFrame()

# 3. INTERFACE PRINCIPAL
st.title("📊 MATI: Painel de Gestão Humanizada (v6.1)")
st.markdown("Monitoramento em tempo real de Atenção, Tensão e Interação no chão de fábrica.")
st.markdown("---")

# O st.empty() cria uma "caixa" vazia na tela que podemos atualizar em loop
placeholder = st.empty()

# 4. LOOP DE ATUALIZAÇÃO EM TEMPO REAL
while True:
    df = carregar_dados()
    
    # Atualiza todo o conteúdo dentro da "caixa" a cada ciclo
    with placeholder.container():
        if df.empty:
            st.warning("⏳ Aguardando dados... Certifique-se de que o script 'camera_mati.py' está rodando, o operador foi identificado e o CSV foi gerado.")
        else:
            # --- SEÇÃO A: KPIs (Indicadores Principais) ---
            # NOVO: Lendo o nome do colaborador da última linha gravada
            colaborador_atual = str(df['colaborador'].iloc[-1])
            
            # Mudamos para 4 colunas para acomodar o nome do operador
            col1, col2, col3, col4 = st.columns(4)
            
            # Extraindo os dados mais recentes (última linha) ou médias
            ultimo_bpm = df['bpm'].iloc[-1]
            ear_medio = df['ear'].mean()
            emocao_predominante = df['emocao'].mode()[0]
            
            with col1:
                st.metric("👤 Operador Atual", colaborador_atual)
            with col2:
                st.metric("💓 Freq. Cardíaca (Atual)", f"{ultimo_bpm} bpm")
            with col3:
                st.metric("👁️ Atenção Média (EAR)", f"{ear_medio:.2f}")
            with col4:
                st.metric("🎭 Humor Predominante", emocao_predominante)
            
            st.markdown("---")
            
            # --- SEÇÃO B: GRÁFICOS DE TENDÊNCIA ---
            col_grafico1, col_grafico2 = st.columns(2)
            
            # Define a coluna 'horario' como o eixo X dos gráficos
            df_grafico = df.set_index('horario')
            
            with col_grafico1:
                st.subheader("Variação de Atenção (Fadiga)")
                st.markdown("Quedas bruscas indicam micro-sonos ou desatenção.")
                st.line_chart(df_grafico['ear'], color="#00a4d6")
                
            with col_grafico2:
                st.subheader("Eletrocardiograma Simulado (BPM)")
                st.markdown("Picos vermelhos podem indicar estresse ou esforço excessivo.")
                st.line_chart(df_grafico['bpm'], color="#ff4b4b")
                
            st.markdown("---")
            
            # --- SEÇÃO C: DISTRIBUIÇÃO DE HUMOR ---
            st.subheader(f"Distribuição de Emoções no Turno: {colaborador_atual}")
            # Conta quantas vezes cada emoção apareceu
            contagem_emocoes = df['emocao'].value_counts()
            # Gera um gráfico de barras verde
            st.bar_chart(contagem_emocoes, color="#4caf50")
            
    # Pausa de 2 segundos antes de ler o CSV novamente
    time.sleep(2)