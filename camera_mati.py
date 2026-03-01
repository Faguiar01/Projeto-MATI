# ============================================================
# PROJETO MATI — Monitoramento Avançado do Trabalhador Inteligente
# Versão 4.1 — Visão + Log CSV + Alerta Sonoro (Windows)
# ============================================================

import cv2
import winsound  # Biblioteca para o bipe no Windows
import csv
from datetime import datetime, timedelta

# --- CONFIGURAÇÕES INICIAIS ---
ARQUIVO_CSV = "dados_mati.csv"
BPM_FIXO = 80
pecas_produzidas = 0

# Classificadores (Devem estar na mesma pasta do script)
classificador_rosto = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificador_olho  = cv2.CascadeClassifier("haarcascade_eye.xml")

# Controle da Trava de Tempo (Anti-Flood)
ultimo_registro = datetime.min
INTERVALO_MINIMO = timedelta(seconds=5)

# --- FUNÇÃO DE PERSISTÊNCIA ---
def salvar_evento_fadiga(horario, bpm, status, pecas):
    try:
        with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow([horario, bpm, status, pecas])
        print(f"[LOG] Evento gravado: {horario} | BIP emitido.")
    except Exception as e:
        print(f"Erro ao salvar no CSV: {e}")

# --- INÍCIO DO SISTEMA ---
cap = cv2.VideoCapture(0)

print("MATI Iniciado. Pressione 'q' para encerrar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    agora = datetime.now()

    # Detecção de Rosto
    rostos = classificador_rosto.detectMultiScale(frame_cinza, 1.1, 5, minSize=(30, 30))
    total_olhos_frame = 0

    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        roi_gray = frame_cinza[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detecção de Olhos dentro do Rosto
        olhos = classificador_olho.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))
        total_olhos_frame += len(olhos)

        for (ex, ey, ew, eh) in olhos:
            cv2.circle(roi_color, (ex + ew//2, ey + eh//2), min(ew, eh)//2, (255, 0, 0), 2)

    # --- LÓGICA DE ALERTA (CRÍTICA) ---
    # Se há rosto mas NÃO detectou olhos
    if len(rostos) > 0 and total_olhos_frame == 0:
        
        # 1. Alerta Visual
        cv2.putText(frame, "ATENCAO: OLHOS NAO DETECTADOS", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. Verificação da Trava (Só apita e grava a cada 5 segundos)
        if (agora - ultimo_registro) > INTERVALO_MINIMO:
            
            # EMITE O SOM (1000Hz por 500ms)
            winsound.Beep(1000, 500) 
            
            # GRAVA NO CSV
            horario_fmt = agora.strftime("%Y-%m-%d %H:%M:%S")
            salvar_evento_fadiga(horario_fmt, BPM_FIXO, "Fadiga Detectada", pecas_produzidas)
            
            # REINICIA O CRONÔMETRO
            ultimo_registro = agora

    cv2.imshow("MATI - Monitoramento de Rosto", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistema encerrado com segurança.")