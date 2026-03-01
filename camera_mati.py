# ============================================================
# PROJETO MATI — camera_mati.py
# Versão 5.2 — MediaPipe via mp.solutions (funciona no Windows)
# ============================================================

import cv2
import csv
import winsound
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial import distance
import mediapipe as mp

# ── CONFIGURAÇÃO DO MEDIAPIPE ────────────────────────────────
# mp.solutions é o caminho correto no mediapipe 0.10.x no Windows
mp_face_mesh = mp.solutions.face_mesh
mp_desenho   = mp.solutions.drawing_utils

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

# ── PARÂMETROS DE FADIGA ─────────────────────────────────────
EAR_LIMIAR           = 0.25
FPS_CAMERA           = 30
FRAMES_PARA_ALERTA   = FPS_CAMERA * 2      # 2 segundos = 60 frames
contador_frames      = 0

# ── CONFIGURAÇÕES DO CSV ─────────────────────────────────────
ARQUIVO_CSV      = "dados_mati.csv"
BPM_FIXO         = 80
pecas_produzidas = 0
ultimo_registro  = datetime.min
INTERVALO_MINIMO = timedelta(seconds=5)

def salvar_evento_fadiga(horario, bpm, status, pecas):
    with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as arquivo:
        csv.writer(arquivo).writerow([horario, bpm, status, pecas])
    print(f"[LOG] {horario} | {status}")

# ── WEBCAM ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: webcam não acessível.")
    exit()

print("MATI v5.2 iniciado! Pressione 'q' para encerrar.")
print("(Warnings do TensorFlow são normais — ignore-os)")

# ── LOOP PRINCIPAL ───────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_mesh.process(frame_rgb)
    ear_medio = None

    if resultado.multi_face_landmarks:
        landmarks = resultado.multi_face_landmarks[0].landmark

        # Extrai coordenadas em pixels dos landmarks dos olhos
        def extrai_pontos(indices):
            return [(int(landmarks[i].x * largura),
                     int(landmarks[i].y * altura)) for i in indices]

        pontos_esq = extrai_pontos(LANDMARKS_OLHO_ESQUERDO)
        pontos_dir = extrai_pontos(LANDMARKS_OLHO_DIREITO)

        ear_medio = (calcula_ear(pontos_esq) + calcula_ear(pontos_dir)) / 2.0

        # Pontos nos olhos (ciano)
        for p in pontos_esq + pontos_dir:
            cv2.circle(frame, p, 2, (255, 255, 0), -1)

        # EAR na tela
        cv2.putText(frame, f"EAR: {ear_medio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Lógica do contador de frames
        if ear_medio < EAR_LIMIAR:
            contador_frames += 1
        else:
            contador_frames = 0

        # Alerta após 2 segundos consecutivos
        if contador_frames >= FRAMES_PARA_ALERTA:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 50), (largura, 120), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "ALERTA: FADIGA DETECTADA!", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            seg = contador_frames / FPS_CAMERA
            cv2.putText(frame, f"Olhos fechados: {seg:.1f}s", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            winsound.Beep(1000, 200)

            agora = datetime.now()
            if (agora - ultimo_registro) > INTERVALO_MINIMO:
                salvar_evento_fadiga(
                    agora.strftime("%Y-%m-%d %H:%M:%S"),
                    BPM_FIXO, "Fadiga Detectada", pecas_produzidas
                )
                ultimo_registro = agora
    else:
        contador_frames = 0
        cv2.putText(frame, "Rosto nao detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Status na base da tela
    if ear_medio is not None:
        cor = (0, 255, 0) if ear_medio >= EAR_LIMIAR else (0, 0, 255)
        txt = "OLHOS: ABERTOS" if ear_medio >= EAR_LIMIAR else "OLHOS: FECHADOS"
        cv2.putText(frame, txt, (10, altura - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

    cv2.imshow("MATI - Monitoramento de Fadiga", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── ENCERRAMENTO ─────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Sistema encerrado com segurança.")