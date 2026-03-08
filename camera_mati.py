# ============================================================
# PROJETO MATI — camera_mati.py
# Versão 8.6 — Terminal Limpo, Auto-Clear e UX de Calibragem
# ============================================================

import os
import time
import warnings

# 1. SILENCIA OS LOGS 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

# 2. PERGUNTA O NOME 
print("="*55)
print(" 🚀 PROJETO MATI - Monitoramento Inteligente 5.0")
print("="*55)
nome_operador = input("🔧 Digite o nome do operador para iniciar o turno: ")

# 3. AUTO-LIMPEZA DO BANCO DE DADOS 
ARQUIVO_CSV = "dados_mati.csv"
if os.path.exists(ARQUIVO_CSV):
    os.remove(ARQUIVO_CSV)
    print("🧹 Banco de dados limpo. Iniciando novo registro...")

print("⏳ Ligando os motores de Inteligência Artificial... Aguarde.")

import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import numpy as np
from deepface import DeepFace
import winsound  

LIMITE_FADIGA = 0.12  
FRAMES_PARA_FADIGA = 10 

TRADUCAO_EMOCOES = {
    'happy': 'Satisfeito', 'sad': 'Fadigado', 'angry': 'Tenso',
    'fear': 'Alerta', 'surprise': 'Surpreso', 'disgust': 'Desconforto', 'neutral': 'Focado' 
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

OLHO_ESQUERDO, OLHO_DIREITO = [362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144]

def calcular_ear(landmarks, indices_olho):
    try:
        p1, p2 = np.array([landmarks[indices_olho[1]].x, landmarks[indices_olho[1]].y]), np.array([landmarks[indices_olho[5]].x, landmarks[indices_olho[5]].y])
        p3, p4 = np.array([landmarks[indices_olho[2]].x, landmarks[indices_olho[2]].y]), np.array([landmarks[indices_olho[4]].x, landmarks[indices_olho[4]].y])
        p5, p6 = np.array([landmarks[indices_olho[0]].x, landmarks[indices_olho[0]].y]), np.array([landmarks[indices_olho[3]].x, landmarks[indices_olho[3]].y])
        return (np.linalg.norm(p1 - p2) + np.linalg.norm(p3 - p4)) / (2.0 * np.linalg.norm(p5 - p6))
    except: return 0.0

def extrair_sinal_testa(frame, face_landmarks, w, h):
    try:
        y_top, y_bottom = int(face_landmarks.landmark[10].y * h), int(face_landmarks.landmark[9].y * h)
        x_left, x_right = int(face_landmarks.landmark[67].x * w), int(face_landmarks.landmark[297].x * w)
        margem_y = int((y_bottom - y_top) * 0.2)
        y_top, y_bottom = max(0, y_top + margem_y), min(h, y_bottom - margem_y)
        x_left, x_right = max(0, x_left), min(w, x_right)
        roi_testa = frame[y_top:y_bottom, x_left:x_right]
        if roi_testa.size == 0: return 0.0, None
        return np.mean(roi_testa[:, :, 1]), (x_left, y_top, x_right, y_bottom)
    except: return 0.0, None

def salvar_log(horario, colab, ear, status, emocao, bpm, sinal_verde):
    if not os.path.exists(ARQUIVO_CSV): pd.DataFrame(columns=['horario', 'colaborador', 'ear', 'status_fadiga', 'emocao', 'bpm', 'sinal_verde']).to_csv(ARQUIVO_CSV, index=False)
    try: pd.DataFrame([{'horario': horario, 'colaborador': colab, 'ear': ear, 'status_fadiga': status, 'emocao': emocao, 'bpm': bpm, 'sinal_verde': sinal_verde}]).to_csv(ARQUIVO_CSV, mode='a', header=False, index=False)
    except: pass

def rodar_camera():
    cap = cv2.VideoCapture(0)
    
    # --- NOVO: TELA DE CALIBRAGEM (UX) ---
    tempo_inicio_calibragem = time.time()
    while time.time() - tempo_inicio_calibragem < 3.5:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        tempo_restante = int(4 - (time.time() - tempo_inicio_calibragem))
        if tempo_restante > 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), 10) # Borda preta de foco
            cv2.putText(frame, f"CALIBRANDO SISTEMA: {tempo_restante}s", (w//2 - 220, h//2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Por favor, olhe para a camera", (w//2 - 160, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("MATI - Sensor Optico", frame)
        cv2.waitKey(1)
    # --------------------------------------

    TEMPO_EMOCAO, ultima_analise, emocao_atual, ultimo_bipe = 1.0, time.time(), "Aguardando...", time.time()
    buffer_verde, bpm_real, contador_olhos_fechados = [], 75, 0 
    
    print("\n🟢 Câmera Ativa. Pressione 'Q' no vídeo para encerrar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_mesh.process(rgb_frame)
        
        ear_medio, sinal_verde, status_atual, cor_status = 0.0, 0.0, "Ativo", (0, 255, 0)

        if resultados.multi_face_landmarks:
            landmarks = resultados.multi_face_landmarks[0]
            ear_medio = (calcular_ear(landmarks.landmark, OLHO_ESQUERDO) + calcular_ear(landmarks.landmark, OLHO_DIREITO)) / 2.0
            
            for pto in OLHO_ESQUERDO + OLHO_DIREITO:
                x, y = int(landmarks.landmark[pto].x * w), int(landmarks.landmark[pto].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if ear_medio < LIMITE_FADIGA:
                contador_olhos_fechados += 1
                if contador_olhos_fechados >= FRAMES_PARA_FADIGA:
                    status_atual, cor_status = "Fadiga", (0, 0, 255)
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15) 
                    cv2.putText(frame, "!!! ALERTA DE FADIGA !!!", (w//2 - 180, h//2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    if time.time() - ultimo_bipe > 1.0:
                        winsound.Beep(2000, 500) 
                        ultimo_bipe = time.time()
            else: contador_olhos_fechados = 0 

            sinal_verde, box_testa = extrair_sinal_testa(frame, landmarks, w, h)
            if box_testa:
                cv2.rectangle(frame, (box_testa[0], box_testa[1]), (box_testa[2], box_testa[3]), (255, 255, 0), 2)
                cv2.putText(frame, "Sensor rPPG", (box_testa[0], box_testa[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                if sinal_verde > 0:
                    buffer_verde.append(sinal_verde)
                    if len(buffer_verde) > 60: 
                        buffer_verde.pop(0)
                        media_sinal = np.mean(buffer_verde)
                        picos = sum(1 for i in range(1, len(buffer_verde)) if buffer_verde[i-1] < media_sinal and buffer_verde[i] >= media_sinal)
                        bpm_real = max(60, min(110, int(picos * 30))) 

            if time.time() - ultima_analise > TEMPO_EMOCAO:
                try: 
                    emocao_ingles = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, silent=True)[0]['dominant_emotion']
                    emocao_atual = TRADUCAO_EMOCOES.get(emocao_ingles, "FOCADO").upper()
                except: emocao_atual = "N/A"
                ultima_analise = time.time()

        salvar_log(datetime.now().strftime("%H:%M:%S"), nome_operador, round(ear_medio, 3), status_atual, emocao_atual, bpm_real, round(sinal_verde, 2))

        cv2.putText(frame, f"EAR: {ear_medio:.2f} ({status_atual})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)
        cv2.putText(frame, f"Estado: {emocao_atual}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(frame, f"BPM (rPPG): {bpm_real}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("MATI - Sensor Optico", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": rodar_camera()