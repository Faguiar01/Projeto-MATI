# ============================================================
# PROJETO MATI — Monitoramento Avançado do Trabalhador Inteligente
# Arquivo: camera_mati.py
# Versão 5.0 — MediaPipe Face Mesh + EAR + Alarme Sonoro + CSV
# ============================================================

import cv2
import csv
import winsound                          # biblioteca nativa do Windows para som
import mediapipe as mp                   # framework de visão computacional do Google
import numpy as np                       # operações matemáticas com arrays/vetores
from datetime import datetime, timedelta
from scipy.spatial import distance       # calcula distância euclidiana entre pontos

# ─────────────────────────────────────────────────────────────
# ETAPA 1 — CONFIGURAÇÃO DO MEDIAPIPE FACE MESH
# ─────────────────────────────────────────────────────────────

# mp.solutions contém todos os módulos prontos do MediaPipe.
# face_mesh é o módulo que detecta 468 pontos precisos no rosto.
mp_face_mesh = mp.solutions.face_mesh

# mp.solutions.drawing_utils oferece funções prontas para desenhar
# os landmarks (pontos) e conexões do rosto na tela.
mp_desenho = mp.solutions.drawing_utils

# FaceMesh() inicializa o modelo de detecção.
# Parâmetros importantes:
#   max_num_faces=1       → monitora apenas 1 trabalhador por vez (mais eficiente)
#   refine_landmarks=True → ativa landmarks de alta precisão ao redor dos olhos e lábios
#                           (necessário para o cálculo EAR funcionar bem)
#   min_detection_confidence → confiança mínima para DETECTAR um rosto pela 1ª vez (0 a 1)
#   min_tracking_confidence  → confiança mínima para RASTREAR o rosto entre frames (0 a 1)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────────────────────────
# ETAPA 2 — ÍNDICES DOS LANDMARKS DOS OLHOS
# ─────────────────────────────────────────────────────────────

# O MediaPipe Face Mesh retorna 468 pontos numerados no rosto.
# Cada número é a posição de um ponto específico (canto do olho, íris, etc.).
# Estes índices foram definidos pelo Google e são fixos para todos os rostos.
# Fonte: https://mediapipe.dev/images/mobile/face_mesh_landmarks.png
#
# Para cada olho precisamos de 6 pontos específicos (os mesmos da fórmula EAR):
#   [0] = canto esquerdo  (p1)
#   [1] = pálpebra superior esquerda  (p2)
#   [2] = pálpebra superior direita   (p3)
#   [3] = canto direito   (p4)
#   [4] = pálpebra inferior direita   (p5)
#   [5] = pálpebra inferior esquerda  (p6)

LANDMARKS_OLHO_ESQUERDO  = [362, 385, 387, 263, 373, 380]
LANDMARKS_OLHO_DIREITO   = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────────────────────
# ETAPA 3 — FUNÇÃO DE CÁLCULO DO EAR
# ─────────────────────────────────────────────────────────────

def calcula_ear(pontos):
    """
    Calcula o Eye Aspect Ratio (EAR) de um olho.
    
    Parâmetro:
        pontos → lista com 6 coordenadas (x, y) dos landmarks do olho,
                 na ordem: [p1, p2, p3, p4, p5, p6]
    
    Retorna:
        ear → número float entre ~0.0 (fechado) e ~0.4 (bem aberto)
    """

    # distance.euclidean() calcula a distância em linha reta entre dois pontos.
    # Matematicamente: √((x2-x1)² + (y2-y1)²)
    # É o "teorema de Pitágoras" aplicado às coordenadas dos pixels.

    # Distâncias VERTICAIS (altura do olho em dois pontos diferentes)
    # p2 está em pontos[1], p6 está em pontos[5]
    dist_vertical_A = distance.euclidean(pontos[1], pontos[5])

    # p3 está em pontos[2], p5 está em pontos[4]
    dist_vertical_B = distance.euclidean(pontos[2], pontos[4])

    # Distância HORIZONTAL (largura total do olho)
    # p1 está em pontos[0], p4 está em pontos[3]
    dist_horizontal  = distance.euclidean(pontos[0], pontos[3])

    # Fórmula EAR: média das alturas dividida pela largura
    # O denominador é multiplicado por 2 para normalizar
    # (temos 2 medidas verticais mas apenas 1 horizontal)
    ear = (dist_vertical_A + dist_vertical_B) / (2.0 * dist_horizontal)

    return ear

# ─────────────────────────────────────────────────────────────
# ETAPA 4 — PARÂMETROS DE FADIGA E CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────

# Limiar EAR: abaixo deste valor consideramos o olho "fechado".
# 0.25 é o valor clássico da literatura científica para detectar
# fechamento de olhos em populações adultas.
# Você pode calibrar este valor conforme o trabalhador monitorado.
EAR_LIMIAR = 0.25

# FPS médio da webcam: usamos para converter segundos em frames.
# Se sua câmera roda a 30fps, 2 segundos = 60 frames consecutivos.
# Ajuste FPS_CAMERA se sua câmera for diferente (ex: 24fps → 48 frames).
FPS_CAMERA = 30
SEGUNDOS_PARA_ALERTA = 2
FRAMES_PARA_ALERTA   = FPS_CAMERA * SEGUNDOS_PARA_ALERTA  # = 60 frames

# Contador de frames consecutivos com olhos fechados.
# Começa em 0 e é incrementado a cada frame onde EAR < limiar.
# Se o trabalhador piscar (olho fechado por ~3 frames), não dispara alerta.
# Se fechar por 60+ frames (~2s), dispara o alerta de fadiga.
contador_frames_fechados = 0

# Configurações do arquivo CSV
ARQUIVO_CSV  = "dados_mati.csv"
BPM_FIXO     = 80
pecas_produzidas = 0

# Trava anti-flood: mesmo mecanismo da versão anterior
ultimo_registro  = datetime.min
INTERVALO_MINIMO = timedelta(seconds=5)

# ─────────────────────────────────────────────────────────────
# ETAPA 5 — FUNÇÃO DE GRAVAÇÃO NO CSV
# ─────────────────────────────────────────────────────────────

def salvar_evento_fadiga(horario, bpm, status, pecas):
    """Acrescenta uma linha de evento de fadiga no CSV sem apagar os dados anteriores."""
    with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow([horario, bpm, status, pecas])
    print(f"[LOG CSV] {horario} | {status} | BPM: {bpm}")

# ─────────────────────────────────────────────────────────────
# ETAPA 6 — INICIALIZAR A WEBCAM
# ─────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Não foi possível acessar a webcam.")
    exit()

print("MATI v5.0 iniciado! Pressione 'q' para encerrar.")

# ─────────────────────────────────────────────────────────────
# ETAPA 7 — LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────

while True:

    ret, frame = cap.read()
    if not ret:
        print("ERRO: Falha ao capturar o frame.")
        break

    # Espelha o frame horizontalmente para parecer um espelho natural.
    # Sem isso, mover a cabeça para a direita move o rosto na tela para a esquerda.
    frame = cv2.flip(frame, 1)

    # Obtém as dimensões do frame para converter coordenadas normalizadas → pixels
    altura, largura, _ = frame.shape

    # O MediaPipe exige imagens em RGB, mas o OpenCV usa BGR por padrão.
    # Convertemos antes de processar e voltamos para BGR ao desenhar.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process() roda o modelo de detecção no frame.
    # Retorna um objeto com todos os landmarks detectados.
    resultado = face_mesh.process(frame_rgb)

    # Variável que guardará o EAR médio dos dois olhos neste frame.
    # Começa como None para sabermos se houve detecção ou não.
    ear_medio = None

    # ── SE DETECTOU ROSTO ────────────────────────────────────
    if resultado.multi_face_landmarks:

        # Pega o primeiro (e único) rosto detectado
        landmarks_rosto = resultado.multi_face_landmarks[0]

        # Extrai as coordenadas (x, y) em PIXELS dos 6 landmarks de cada olho.
        #
        # landmark.x e landmark.y são valores NORMALIZADOS entre 0.0 e 1.0.
        # (0,0) = canto superior esquerdo da imagem
        # (1,1) = canto inferior direito da imagem
        #
        # Para obter pixels reais multiplicamos pelo tamanho do frame:
        #   pixel_x = landmark.x * largura_em_pixels
        #   pixel_y = landmark.y * altura_em_pixels
        #
        # int() arredonda para o pixel mais próximo (coordenadas são inteiras)
        def extrai_pontos(indices):
            return [
                (
                    int(landmarks_rosto.landmark[i].x * largura),
                    int(landmarks_rosto.landmark[i].y * altura)
                )
                for i in indices
            ]

        pontos_olho_esq = extrai_pontos(LANDMARKS_OLHO_ESQUERDO)
        pontos_olho_dir = extrai_pontos(LANDMARKS_OLHO_DIREITO)

        # Calcula o EAR de cada olho individualmente
        ear_esq = calcula_ear(pontos_olho_esq)
        ear_dir = calcula_ear(pontos_olho_dir)

        # EAR médio dos dois olhos: mais robusto que usar apenas um,
        # pois compensa leves inclinações de cabeça que afetam um lado só.
        ear_medio = (ear_esq + ear_dir) / 2.0

        # Desenha pontos nos landmarks dos olhos (visualização)
        for ponto in pontos_olho_esq + pontos_olho_dir:
            # Círculo pequeno (raio=2, preenchido com thickness=-1) em ciano
            cv2.circle(frame, ponto, 2, (255, 255, 0), -1)

        # Exibe o valor EAR atual na tela (canto superior esquerdo)
        cv2.putText(frame, f"EAR: {ear_medio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ── LÓGICA DO CONTADOR DE FRAMES ─────────────────────
        #
        # Se EAR cair abaixo do limiar, incrementamos o contador.
        # Se EAR voltar ao normal (piscada terminou), zeramos o contador.
        # O alerta só dispara quando o contador ultrapassar FRAMES_PARA_ALERTA.
        #
        # Fluxo visual:
        #
        #  EAR:  .32  .31  .30  .14  .13  .12  .13  .30  .31
        #        OPEN OPEN OPEN SHUT SHUT SHUT SHUT OPEN OPEN
        # Cont:   0    0    0    1    2    3    4    0    0
        #                              ↑ nunca chega em 60 → sem alerta (piscada normal)

        if ear_medio < EAR_LIMIAR:
            contador_frames_fechados += 1
        else:
            # Olhos abertos novamente: reseta o contador
            contador_frames_fechados = 0

        # ── ALERTA DE FADIGA (só após 2 segundos consecutivos) ──
        if contador_frames_fechados >= FRAMES_PARA_ALERTA:

            # Retângulo vermelho semitransparente como fundo do alerta
            # Criamos uma cópia para aplicar transparência com addWeighted
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 50), (largura, 120), (0, 0, 180), -1)
            # addWeighted mistura o overlay com o frame original
            # 0.4 = 40% da cor do retângulo + 60% da imagem original
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Texto do alerta em vermelho brilhante
            cv2.putText(frame, "ALERTA: FADIGA DETECTADA!", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # Exibe há quantos frames o olho está fechado (para debug)
            segundos_fechado = contador_frames_fechados / FPS_CAMERA
            cv2.putText(frame, f"Olhos fechados: {segundos_fechado:.1f}s",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # ── ALARME SONORO ────────────────────────────────
            # winsound.Beep(frequência_Hz, duração_ms)
            # 1000 Hz por 200ms: bipe curto e agudo, não intrusivo
            # O alarme toca a cada frame de alerta — considere usar
            # threading para não bloquear o loop se necessário.
            winsound.Beep(1000, 200)

            # ── GRAVAÇÃO NO CSV (com trava de 5 segundos) ────
            agora = datetime.now()
            if (agora - ultimo_registro) > INTERVALO_MINIMO:
                horario_fmt = agora.strftime("%Y-%m-%d %H:%M:%S")
                salvar_evento_fadiga(
                    horario=horario_fmt,
                    bpm=BPM_FIXO,
                    status="Fadiga Detectada",
                    pecas=pecas_produzidas
                )
                ultimo_registro = agora

    else:
        # Nenhum rosto detectado neste frame: zera o contador
        contador_frames_fechados = 0
        cv2.putText(frame, "Rosto nao detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # ── INDICADOR VISUAL DO LIMIAR ────────────────────────────
    # Barra de status no canto inferior: verde se aberto, vermelho se fechado
    if ear_medio is not None:
        cor_status = (0, 255, 0) if ear_medio >= EAR_LIMIAR else (0, 0, 255)
        cv2.putText(frame, "STATUS: ATIVO", (10, altura - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)

    cv2.imshow("MATI - Monitoramento de Rosto v5", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- ENCERRAMENTO SEGURO ---
cap.release()
cv2.destroyAllWindows()
print("Câmera liberada. Sistema encerrado com segurança.")