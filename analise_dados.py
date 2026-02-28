# ============================================================
# PROJETO MATI — Monitoramento Avançado do Trabalhador Inteligente
# Arquivo: analise_dados.py
# Objetivo: Ler e analisar os dados do arquivo dados_mati.csv
# ============================================================

import pandas as pd

# --- ETAPA 1: LER O ARQUIVO CSV ---
df = pd.read_csv("dados_mati.csv")

# --- ETAPA 2: VISUALIZAR AS PRIMEIRAS LINHAS ---
print("=" * 50)
print("PRÉVIA DOS DADOS — PRIMEIRAS 5 LINHAS:")
print("=" * 50)
print(df.head(5))

# --- ETAPA 3: CALCULAR AS MÉTRICAS DO DIA ---
print("\n" + "=" * 50)
print("ANÁLISE DO DIA:")
print("=" * 50)

# Calcula a média da coluna BPM e o total de Peças Produzidas
media_bpm = round(df["BPM"].mean(), 1)
total_pecas = df["Pecas_Produzidas"].sum()

# Imprime os resultados na tela
print(f"Média de BPM no dia:         {media_bpm} bpm")
print(f"Total de Peças Produzidas:   {total_pecas} peças")
print("=" * 50)
print("Análise concluída com sucesso!")
print("=" * 50)