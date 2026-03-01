# ============================================================
# PROJETO MATI — camera_mati.py
# Versão 5.3 — Gravação Contínua para Dashboard
# ============================================================

import cv2
import csv
import winsound
import numpy as np
import os
from datetime import datetime, timedelta
from scipy.spatial import distance
import mediapipe as mp

# ── CONFIGURAÇÃO DO MEDIAPIPE ────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── ÍNDICES DOS LANDMARKS DOS OLHOS ─────────────────────────
LANDMARKS_OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]
LANDMARKS_OLHO_DIREITO  = [33,  160, 158, 133, 153, 144]

# ── FUNÇÃO EAR ───────────────────────────────────────────────
def calcula_ear(pontos):
    A = distance.euclidean(pontos[1], pontos[5])
    B = distance.euclidean(pontos[2], pontos[4])
    C = distance.euclidean(pontos[0], pontos[3])
    return (A + B) / (2.0 * C)

# ── PARÂMETROS E ESTADOS ─────────────────────────────────────
EAR_LIMIAR           = 0.25
FPS_CAMERA           = 30
FRAMES_PARA_ALERTA   = FPS_CAMERA * 2
contador_frames      = 0
ARQUIVO_CSV          = "dados_mati.csv"

# Inicializa o CSV com cabeçalho (se não existir)
if not os.path.exists(ARQUIVO_CSV):
    with open(ARQUIVO_CSV, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["horario", "ear", "status"])

def salvar_log_continuo(horario, ear, status):
    with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([horario, ear, status])

# ── WEBCAM ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("MATI v5.3 iniciado! Gerando dados para o Dashboard...")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_mesh.process(frame_rgb)
    
    ear_medio = 0.0
    status_atual = "Normal"

    if resultado.multi_face_landmarks:
        landmarks = resultado.multi_face_landmarks[0].landmark

        def extrai_pontos(indices):
            return [(int(landmarks[i].x * largura), int(landmarks[i].y * altura)) for i in indices]

        pontos_esq = extrai_pontos(LANDMARKS_OLHO_ESQUERDO)
        pontos_dir = extrai_pontos(LANDMARKS_OLHO_DIREITO)
        ear_medio = (calcula_ear(pontos_esq) + calcula_ear(pontos_dir)) / 2.0

        # Desenho dos pontos
        for p in pontos_esq + pontos_dir:
            cv2.circle(frame, p, 2, (255, 255, 0), -1)

        # Lógica de Alerta
        if ear_medio < EAR_LIMIAR:
            contador_frames += 1
            status_atual = "Fadiga"
            if contador_frames >= FRAMES_PARA_ALERTA:
                cv2.putText(frame, "ALERTA DE FADIGA!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                winsound.Beep(1000, 100)
        else:
            contador_frames = 0

        # SALVAMENTO CONTÍNUO (Essencial para o Dashboard)
        salvar_log_continuo(datetime.now().strftime("%H:%M:%S"), round(ear_medio, 3), status_atual)

    # UI básica
    cv2.putText(frame, f"EAR: {ear_medio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("MATI - Monitoramento", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()