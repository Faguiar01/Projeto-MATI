# ============================================================
# PROJETO MATI — camera_mati.py
# Versão 6.2 — Inicialização Limpa e Otimizada
# ============================================================

import os
# Resolve conflito interno de bibliotecas de IA (TensorFlow/MediaPipe)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Silencia os logs de inicialização mais chatos do TensorFlow e MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'

# ── 0. CAPTURA DE DADOS DO USUÁRIO (PRIMEIRA COISA A RODAR) ─────
print("="*50)
print("MATI v6.2 - Sistema de Inicialização")
print("="*50)

# Captura o nome ANTES de carregar as IAs pesadas
nome_colaborador = input("Digite o nome do operador para iniciar o turno: ").strip()
if not nome_colaborador:
    nome_colaborador = "Operador_Padrao"

print(f"\n✅ Operador {nome_colaborador} registrado.")
print("Carregando motores de Inteligência Artificial... (Pode levar alguns segundos)")
print("-" * 50)

# ── 1. IMPORTAÇÕES PESADAS ──────────────────────────────────────
import cv2
import csv
import winsound
import numpy as np
import time
from datetime import datetime
from scipy.spatial import distance
from collections import Counter
import mediapipe as mp
from deepface import DeepFace

# ── 2. CONFIGURAÇÃO DO MEDIAPIPE (FADIGA FÍSICA) ────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Índices dos pontos (landmarks) dos olhos na malha do MediaPipe
LANDMARKS_OLHO_ESQUERDO = [362, 385, 387, 263, 373, 380]
LANDMARKS_OLHO_DIREITO  = [33,  160, 158, 133, 153, 144]

# Função matemática para calcular a proporção de abertura do olho (EAR)
def calcula_ear(pontos):
    A = distance.euclidean(pontos[1], pontos[5])
    B = distance.euclidean(pontos[2], pontos[4])
    C = distance.euclidean(pontos[0], pontos[3])
    return (A + B) / (2.0 * C)

# ── 3. PARÂMETROS GERAIS E DICIONÁRIO ───────────────────────────
EAR_LIMIAR           = 0.25
FPS_CAMERA           = 30
FRAMES_PARA_ALERTA   = FPS_CAMERA * 2
contador_frames      = 0
ARQUIVO_CSV          = "dados_mati.csv"

# Dicionário de Tradução para termos da Indústria 5.0
traducao_emocoes = {
    'neutral': 'Concentrado', 'happy': 'Satisfeito', 'fear': 'Tensao/Estresse',
    'angry': 'Irritacao', 'sad': 'Desanimado', 'surprise': 'Surpreso', 'disgust': 'Desconforto'
}

# Configuração do DeepFace e Filtro de Ruído
INTERVALO_EMOCAO = 0.5
ultimo_tempo_emocao = time.time()
historico_emocoes = []
JANELA_ESTABILIZACAO = 8
emocao_final = "Analisando..."

# Cria o arquivo CSV e o cabeçalho se ele não existir
if not os.path.exists(ARQUIVO_CSV):
    with open(ARQUIVO_CSV, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["horario", "colaborador", "ear", "status_fadiga", "emocao", "bpm"])

def salvar_log_dashboard(horario, colaborador, ear, status_fadiga, emocao, bpm):
    with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([horario, colaborador, ear, status_fadiga, emocao, bpm])

# ── 4. INÍCIO DA CAPTURA DA WEBCAM ──────────────────────────────
cap = cv2.VideoCapture(0)

print("\n🚀 Câmera Ativa! Pressione 'q' na janela do vídeo para sair.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- A. PROCESSAMENTO DE FADIGA (MEDIAPIPE) E BOUNDING BOX ---
    resultado = face_mesh.process(frame_rgb)
    ear_medio = 0.0
    status_atual = "Ativo"

    if resultado.multi_face_landmarks:
        landmarks = resultado.multi_face_landmarks[0].landmark

        # CÁLCULO E DESENHO DO BOUNDING BOX
        x_min, y_min = largura, altura
        x_max, y_max = 0, 0

        for lm in landmarks:
            x, y = int(lm.x * largura), int(lm.y * altura)
            if x < x_min: x_min = x
            if y < y_min: y_min = y
            if x > x_max: x_max = x
            if y > y_max: y_max = y

        padding = 20
        cv2.rectangle(frame, (x_min - padding, y_min - padding), 
                      (x_max + padding, y_max + padding), (0, 255, 0), 2)
        cv2.putText(frame, f"MATI: {nome_colaborador}", (x_min - padding, y_min - padding - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # CÁLCULO EAR (OLHOS)
        def extrai_pontos(indices):
            return [(int(landmarks[i].x * largura), int(landmarks[i].y * altura)) for i in indices]

        pontos_esq = extrai_pontos(LANDMARKS_OLHO_ESQUERDO)
        pontos_dir = extrai_pontos(LANDMARKS_OLHO_DIREITO)
        
        ear_medio = (calcula_ear(pontos_esq) + calcula_ear(pontos_dir)) / 2.0

        if ear_medio < EAR_LIMIAR:
            contador_frames += 1
            status_atual = "Fadiga"
            if contador_frames >= FRAMES_PARA_ALERTA:
                cv2.putText(frame, "ALERTA: FADIGA!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                winsound.Beep(1000, 100)
        else:
            contador_frames = 0
            status_atual = "Ativo"

        for p in pontos_esq + pontos_dir:
            cv2.circle(frame, p, 2, (255, 255, 0), -1)

    # --- B. PROCESSAMENTO DE EMOÇÃO COM ESTABILIZAÇÃO ---
    tempo_agora = time.time()
    if tempo_agora - ultimo_tempo_emocao >= INTERVALO_EMOCAO:
        try:
            analise = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
            emocao_raw = analise[0]['dominant_emotion']
            emocao_traduzida = traducao_emocoes.get(emocao_raw, emocao_raw)
            historico_emocoes.append(emocao_traduzida)
        except:
            pass
        
        if len(historico_emocoes) > 0:
            emocao_final = Counter(historico_emocoes).most_common(1)[0][0]
            if len(historico_emocoes) >= JANELA_ESTABILIZACAO:
                historico_emocoes = historico_emocoes[4:]
        
        ultimo_tempo_emocao = time.time()

    # --- C. MOTOR DE BPM SIMULADO ---
    bpm_base = 75
    if emocao_final == 'Tensao/Estresse' or emocao_final == 'Irritacao': 
        bpm_base = 95
    elif emocao_final == 'Concentrado': 
        bpm_base = 70
        
    bpm_simulado = bpm_base + np.random.randint(-2, 3)

    # --- D. SALVAMENTO NO CSV ---
    salvar_log_dashboard(
        datetime.now().strftime("%H:%M:%S"), 
        nome_colaborador, 
        round(ear_medio, 3), 
        status_atual, 
        emocao_final,
        bpm_simulado
    )

    # --- E. INTERFACE VISUAL (OVERLAY) ---
    cv2.putText(frame, f"Atencao (EAR): {ear_medio:.2f} ({status_atual})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cor_humor = (255, 255, 0)
    if emocao_final == 'Satisfeito': cor_humor = (0, 255, 0)
    if emocao_final in ['Tensao/Estresse', 'Irritacao', 'Desanimado']: cor_humor = (0, 0, 255)

    cv2.putText(frame, f"Humor: {emocao_final}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_humor, 2)
    cv2.putText(frame, f"BPM Fisiologico: {bpm_simulado} bpm", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("MATI v6.2 - Chao de Fabrica", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()