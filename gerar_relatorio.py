# ============================================================
# PROJETO MATI — gerar_relatorio.py
# Versão 8.5 — Laudo Calibrado (Menos falso-positivo de estresse)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def criar_relatorio():
    if not os.path.exists('dados_mati.csv'): return

    try:
        df = pd.read_csv('dados_mati.csv')
        df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce')
        df = df.dropna(subset=['horario'])
        fadiga = df[df['status_fadiga'] == 'Fadiga'].copy()
        
        # Gráfico Duplo (BPM + Fadiga)
        fig, eixo1 = plt.subplots(figsize=(10, 4.5))
        fig.suptitle('Analise Fisiologica e Comportamental — MATI', fontsize=12, fontweight='bold')
        
        eixo1.plot(df['horario'], df['bpm'], color='steelblue', label='BPM (Coracao)', linewidth=2, alpha=0.7)
        eixo1.set_ylabel('BPM', color='steelblue', fontsize=10)
        eixo1.tick_params(axis='x', rotation=45)
        
        eixo2 = eixo1.twinx()
        if not fadiga.empty: eixo2.scatter(fadiga['horario'], fadiga['bpm'], color='red', label='Fadiga Detectada', zorder=5, s=60)
        eixo2.set_ylabel('Eventos de Fadiga', color='red', fontsize=10)
        
        linhas1, labels1 = eixo1.get_legend_handles_labels()
        linhas2, labels2 = eixo2.get_legend_handles_labels()
        eixo1.legend(linhas1 + linhas2, labels1 + labels2, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig('relatorio_fadiga_mati.png', dpi=150, bbox_inches='tight')
        plt.close()

        # KPIs
        operador = str(df['colaborador'].iloc[-1]).upper()
        ear_medio, bpm_medio = df['ear'].mean(), df['bpm'].mean()
        emocao_pred = str(df['emocao'].mode()[0]).upper()
        total_alertas, total_registros = len(fadiga), len(df)
        porcentagem_fadiga = (total_alertas / total_registros) * 100 if total_registros > 0 else 0

        # NOVA CALIBRAGEM DO VEREDITO (Mais inteligente)
        if porcentagem_fadiga > 15:
            veredito, feedback = "RISCO ALTO: FADIGA OPERACIONAL", "Colaborador apresentou multiplos sinais de sonolencia e quebra de atencao. Recomendada pausa imediata."
        elif bpm_medio > 100 and emocao_pred in ['TENSO', 'ALERTA', 'FADIGADO']:
            veredito, feedback = "ALERTA: SOBRECARGA / ESTRESSE", "Indicadores apontam tensao comportamental ALINHADA com aceleracao cardiaca. Sugere-se verificacao ergonomica."
        elif emocao_pred in ['TENSO', 'ALERTA', 'FADIGADO']:
            veredito, feedback = "ATENCAO: TENSAO COMPORTAMENTAL", "O colaborador demonstrou foco tenso. Pode indicar dificuldade visual, ma iluminacao ou desconforto postural."
        else:
            veredito, feedback = "APTO: OPERACAO SEGURA", "Colaborador demonstra bons niveis de atencao, estabilidade e foco comportamental. Apto para atividades."

        # Construção do PDF
        pdf = FPDF()
        pdf.add_page()
        
        if os.path.exists("logo_mati.png"):
            pdf.image("logo_mati.png", x=10, y=8, w=35)
            pdf.ln(30)
        else:
            pdf.ln(10)

        pdf.set_font("Arial", 'B', 15)
        pdf.cell(0, 8, txt="MATI - Laudo de Seguranca Operacional", ln=True, align='C')
        pdf.set_font("Arial", size=9)
        pdf.cell(0, 5, txt="Gestao Humanizada - Industria 5.0", ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, txt=f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}   |   Operador Analisado: {operador}", ln=True)
        pdf.ln(2)
        pdf.image('relatorio_fadiga_mati.png', x=10, w=190)
        pdf.ln(3)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt="1. Indicadores Chave de Desempenho (KPIs):", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 6, txt=f"- Nivel de Atencao Medio (EAR): {ear_medio:.2f} (Ideal > 0.15)", ln=True)
        pdf.cell(0, 6, txt=f"- Freq. Cardiaca Media (rPPG): {bpm_medio:.0f} BPM", ln=True)
        pdf.cell(0, 6, txt=f"- Estado Comportamental Predominante: {emocao_pred}", ln=True)
        pdf.cell(0, 6, txt=f"- Quebras de Atencao (Olhos Fechados > 1s): {total_alertas}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt="2. Veredito do Sistema:", ln=True)
        pdf.set_font("Arial", 'B', 11)
        if "RISCO ALTO" in veredito: pdf.set_text_color(200, 0, 0)
        elif "ALERTA" in veredito or "ATENCAO" in veredito: pdf.set_text_color(200, 150, 0)
        else: pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 6, txt=f">> {veredito} <<", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt="3. Diretriz para o Gestor:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, txt=feedback)
        
        pdf.output("Laudo_MATI.pdf")

    except Exception as e: print(f"Erro: {e}")

if __name__ == "__main__": criar_relatorio()