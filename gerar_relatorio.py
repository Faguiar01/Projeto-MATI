# ============================================================
# PROJETO MATI — Relatório de Fadiga
# Arquivo: gerar_relatorio.py
# Versão melhorada com BPM, volume por hora e exportação PNG
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

try:
    # --- LEITURA E PREPARAÇÃO DOS DADOS ---
    df = pd.read_csv('dados_mati.csv')
    df['Horario'] = pd.to_datetime(df['Horario'], errors='coerce')

    # Remove linhas onde a conversão de data falhou (NaT)
    df = df.dropna(subset=['Horario'])

    # Filtra apenas eventos de fadiga
    fadiga = df[df['Status_Rosto'] == 'Fadiga Detectada'].copy()

    if fadiga.empty:
        print("Nenhum dado de fadiga encontrado no CSV.")
    else:

        # --- RESUMO TEXTUAL NO TERMINAL ---
        total_alertas   = len(fadiga)
        media_bpm       = round(df['BPM'].mean(), 1)
        primeiro_alerta = fadiga['Horario'].min().strftime('%H:%M:%S')
        ultimo_alerta   = fadiga['Horario'].max().strftime('%H:%M:%S')

        print("=" * 45)
        print("       RELATÓRIO DE FADIGA — MATI")
        print("=" * 45)
        print(f"  Total de alertas registrados : {total_alertas}")
        print(f"  Média de BPM no período      : {media_bpm} bpm")
        print(f"  Primeiro alerta              : {primeiro_alerta}")
        print(f"  Último alerta                : {ultimo_alerta}")
        print("=" * 45)

        # --- CONSTRUÇÃO DO GRÁFICO DUPLO ---

        # figsize define largura x altura em polegadas
        fig, eixo1 = plt.subplots(figsize=(12, 6))
        fig.suptitle('Relatório de Fadiga — Projeto MATI', fontsize=14, fontweight='bold')

        # Eixo 1 (esquerda): linha de BPM ao longo do tempo
        eixo1.plot(df['Horario'], df['BPM'],
                   color='steelblue', label='BPM', linewidth=2, alpha=0.7)
        eixo1.set_ylabel('BPM', color='steelblue', fontsize=11)
        eixo1.set_xlabel('Horário', fontsize=11)
        eixo1.tick_params(axis='y', labelcolor='steelblue')

        # Eixo 2 (direita): pontos de alerta de fadiga
        # twinx() cria um segundo eixo Y que compartilha o mesmo eixo X
        eixo2 = eixo1.twinx()
        eixo2.scatter(fadiga['Horario'], fadiga['BPM'],
                      color='red', label='Fadiga Detectada',
                      zorder=5, s=80)  # s=80 é o tamanho do ponto
        eixo2.set_ylabel('Eventos de Fadiga (BPM no momento)', color='red', fontsize=11)
        eixo2.tick_params(axis='y', labelcolor='red')

        # Combina as legendas dos dois eixos em uma só caixa
        linhas1, labels1 = eixo1.get_legend_handles_labels()
        linhas2, labels2 = eixo2.get_legend_handles_labels()
        eixo1.legend(linhas1 + linhas2, labels1 + labels2, loc='upper left')

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()  # Ajusta margens automaticamente para nada ser cortado

        # Salva como imagem PNG para uso em relatórios
        plt.savefig('relatorio_fadiga_mati.png', dpi=150, bbox_inches='tight')
        print("Gráfico salvo como 'relatorio_fadiga_mati.png'")
        plt.show()

except FileNotFoundError:
    # Erro específico para arquivo não encontrado — mais informativo que Exception genérico
    print("ERRO: arquivo 'dados_mati.csv' não encontrado na pasta.")
except Exception as e:
    print(f"Erro inesperado: {e}")