# ============================================================
# PROJETO MATI — camera_mati.py
# Versão 5.4 — Unificação (Com Correção de Protobuf)
# ============================================================

import os
# --- CORREÇÃO DO CONFLITO DE BIBLIOTECAS (MediaPipe vs TensorFlow) ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import csv
import winsound
import numpy as np
import time
from datetime import datetime
from scipy.spatial import distance
import mediapipe as mp

# Importamos o motor de emoções
from deepface import DeepFace

# ── CONFIGURAÇÃO DO MEDIAPIPE (FADIGA FÍSICA) ────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LANDMARKS_OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]
LANDMARKS_OLHO_DIREITO  = [33,  160, 158, 133, 153, 144]

def calcula_ear(pontos):
    A = distance.euclidean(pontos[1], pontos[5])
    B = distance.euclidean(pontos[2], pontos[4])
    C = distance.euclidean(pontos[0], pontos[3])
    return (A + B) / (2.0 * C)

# ── PARÂMETROS GERAIS E BANCO DE DADOS ───────────────────────
EAR_LIMIAR           = 0.25
FPS_CAMERA           = 30
FRAMES_PARA_ALERTA   = FPS_CAMERA * 2
contador_frames      = 0
ARQUIVO_CSV          = "dados_mati.csv"

# ── PARÂMETROS DO DEEPFACE (ESTRESSE/EMOÇÃO) ─────────────────
INTERVALO_EMOCAO = 1.5 # Roda a IA pesada a cada 1.5 segundos
ultimo_tempo_emocao = time.time()
emocao_atual = "Analisando..."

# Inicializa o CSV com a NOVA COLUNA de Emoção
if not os.path.exists(ARQUIVO_CSV):
    with open(ARQUIVO_CSV, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["horario", "ear", "status_fadiga", "emocao"])

def salvar_log_dashboard(horario, ear, status_fadiga, emocao):
    with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([horario, ear, status_fadiga, emocao])

# ── WEBCAM ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

print("MATI v5.4 iniciado! Câmera Dupla (EAR + Emoção) rodando...")
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. PROCESSAMENTO GEOMÉTRICO (MEDIAPIPE)
    resultado = face_mesh.process(frame_rgb)
    ear_medio = 0.0
    status_atual = "Ativo"

    if resultado.multi_face_landmarks:
        landmarks = resultado.multi_face_landmarks[0].landmark

        def extrai_pontos(indices):
            return [(int(landmarks[i].x * largura), int(landmarks[i].y * altura)) for i in indices]

        pontos_esq = extrai_pontos(LANDMARKS_OLHO_ESQUERDO)
        pontos_dir = extrai_pontos(LANDMARKS_OLHO_DIREITO)
        ear_medio = (calcula_ear(pontos_esq) + calcula_ear(pontos_dir)) / 2.0

        if ear_medio < EAR_LIMIAR:
            contador_frames += 1
            status_atual = "Fadiga"
            if contador_frames >= FRAMES_PARA_ALERTA:
                cv2.putText(frame, "ALERTA: FADIGA!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                winsound.Beep(1000, 100)
        else:
            contador_frames = 0
            status_atual = "Ativo"

        for p in pontos_esq + pontos_dir:
            cv2.circle(frame, p, 2, (255, 255, 0), -1)

    # 2. PROCESSAMENTO DE EMOÇÃO (DEEPFACE)
    tempo_agora = time.time()
    if tempo_agora - ultimo_tempo_emocao >= INTERVALO_EMOCAO:
        try:
            analise = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
            emocao_atual = analise[0]['dominant_emotion']
        except:
            emocao_atual = "Desconhecido"
        
        ultimo_tempo_emocao = time.time()

    # 3. SALVAMENTO UNIFICADO NO CSV
    salvar_log_dashboard(
        datetime.now().strftime("%H:%M:%S"), 
        round(ear_medio, 3), 
        status_atual, 
        emocao_atual.upper()
    )

    # 4. INTERFACE VISUAL (OVERLAY)
    cv2.putText(frame, f"EAR: {ear_medio:.2f} ({status_atual})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cor_emocao = (0, 255, 0)
    if emocao_atual in ['angry', 'fear', 'sad']: cor_emocao = (0, 0, 255)
    elif emocao_atual == 'happy': cor_emocao = (0, 255, 0)
    else: cor_emocao = (255, 255, 0)

    cv2.putText(frame, f"Humor: {emocao_atual.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_emocao, 2)

    cv2.imshow("MATI - Monitoramento 5.0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()