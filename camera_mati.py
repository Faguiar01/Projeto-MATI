# ============================================================
# PROJETO MATI — Monitoramento Avançado do Trabalhador Inteligente
# Arquivo: camera_mati.py
# Versão 3.0 — Detecção de Rosto + Olhos + Contador em Tempo Real
# ============================================================

import cv2

# --- ETAPA 1: CARREGAR OS DOIS CLASSIFICADORES ---

# Classificador de rosto (já conhecido da versão anterior)
classificador_rosto = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Classificador de olhos: mesmo princípio do de rosto, mas treinado
# com padrões específicos de olhos (formato amendoado, região escura da íris, etc.)
classificador_olho = cv2.CascadeClassifier("haarcascade_eye.xml")

# Verificações de segurança para os dois arquivos
if classificador_rosto.empty():
    print("ERRO: haarcascade_frontalface_default.xml não encontrado.")
    exit()

if classificador_olho.empty():
    print("ERRO: haarcascade_eye.xml não encontrado.")
    exit()

print("Classificadores carregados com sucesso!")

# --- ETAPA 2: CONECTAR À WEBCAM ---

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Não foi possível acessar a webcam.")
    exit()

print("Câmera iniciada! Pressione 'q' para encerrar.")

# --- ETAPA 3: LOOP PRINCIPAL ---

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERRO: Falha ao capturar o frame.")
        break

    # Converte o frame completo para cinza (necessário para os classificadores)
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame inteiro (igual à versão anterior)
    rostos_detectados = classificador_rosto.detectMultiScale(
        frame_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Variável que vai acumular a quantidade de olhos encontrados neste frame.
    # Começa em 0 a cada frame para sempre refletir o estado atual.
    total_olhos_frame = 0

    # --- ETAPA 4: PARA CADA ROSTO, CRIAR A ROI E BUSCAR OLHOS ---

    for (x, y, w, h) in rostos_detectados:

        # Desenha o retângulo verde ao redor do rosto (mantido da versão anterior)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- [NOVO] CRIANDO A ROI — REGIÃO DE INTERESSE ---
        #
        # CONCEITO FUNDAMENTAL: Por que não buscar os olhos no frame inteiro?
        #
        # Imagine que o frame tem 640x480 = 307.200 pixels para analisar.
        # Um rosto detectado tem talvez 150x150 = 22.500 pixels.
        # Buscar olhos só dentro do rosto significa analisar 22.500 pixels
        # em vez de 307.200 — isso é ~14x menos trabalho para o processador.
        #
        # Além disso, evita "falsos positivos": sem a ROI, o classificador
        # de olhos poderia detectar padrões semelhantes a olhos em outros
        # objetos do cenário (botões de camisa, luminárias, etc.).
        #
        # COMO A ROI FUNCIONA TECNICAMENTE:
        # Um frame no OpenCV é uma matriz numpy (grade de números).
        # Cada pixel é acessado por [linha, coluna] = [y, x].
        # A notação [y : y+h, x : x+w] é um "fatiamento" (slice) dessa matriz,
        # como recortar um pedaço de uma planilha Excel.
        # Não criamos uma cópia: roi_gray e roi_color são "janelas"
        # que apontam para a MESMA memória do frame original.
        # Então quando desenhamos na roi_color, o frame original é atualizado.

        # roi_gray: recorte em CINZA da área do rosto.
        # Usado como ENTRADA para o classificador detectar os olhos
        # (o classificador só aceita imagens em escala de cinza).
        # [y : y+h] → seleciona as LINHAS  do topo até a base do rosto
        # [x : x+w] → seleciona as COLUNAS da esquerda até a direita do rosto
        roi_gray = frame_cinza[y : y + h, x : x + w]

        # roi_color: recorte COLORIDO da mesma área do rosto.
        # Usado como SAÍDA para desenhar os círculos dos olhos com cor.
        # Como é uma "janela" do frame original, desenhar aqui
        # aparece automaticamente no frame que será exibido na tela.
        roi_color = frame[y : y + h, x : x + w]

        # --- [NOVO] DETECTAR OLHOS DENTRO DA ROI ---

        # Mesmo método detectMultiScale, mas agora aplicado à roi_gray
        # (que é apenas a região do rosto, não o frame inteiro).
        # scaleFactor menor (1.05) pois os olhos são menores e precisam
        # de uma varredura mais fina para serem encontrados com precisão.
        # minSize menor (20x20) pois olhos ocupam menos pixels que um rosto.
        olhos_detectados = classificador_olho.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(20, 20)
        )

        # Acumula a contagem de olhos encontrados neste rosto
        total_olhos_frame += len(olhos_detectados)

        # --- [NOVO] DESENHAR CÍRCULOS NOS OLHOS DETECTADOS ---

        # Para cada olho, recebemos suas coordenadas RELATIVAS à ROI.
        # ex: se o rosto começa em x=100 no frame, e o olho está em ex=30
        # dentro da ROI, o olho está em x=130 no frame original.
        # Mas como roi_color já é a janela correta do frame, usamos
        # as coordenadas diretamente sem precisar somar x e y manualmente.
        for (ex, ey, ew, eh) in olhos_detectados:

            # Calcula o centro do círculo:
            # Centro X = posição X do olho + metade da largura
            # Centro Y = posição Y do olho + metade da altura
            centro_x = ex + ew // 2  # // é divisão inteira (sem decimais)
            centro_y = ey + eh // 2

            # Calcula o raio como metade da menor dimensão do retângulo do olho
            # para que o círculo caiba bem dentro do espaço detectado
            raio = min(ew, eh) // 2

            # cv2.circle() desenha um círculo sobre a roi_color.
            # Parâmetros:
            #   roi_color        → imagem onde desenhar (janela do frame original)
            #   (centro_x, centro_y) → ponto central do círculo
            #   raio             → raio em pixels
            #   (255, 0, 0)      → cor em BGR: Azul puro
            #   2                → espessura da linha (-1 preencheria o círculo)
            cv2.circle(roi_color, (centro_x, centro_y), raio, (255, 0, 0), 2)

    # --- ETAPA 5: EXIBIR O CONTADOR DE OLHOS NA TELA ---

    # Monta o texto do contador com o total de olhos encontrados neste frame
    texto_contador = f"Olhos Detectados: {total_olhos_frame}"

    # cv2.putText() para escrever o contador no canto superior esquerdo da tela.
    # A posição (10, 30) significa: 10px da borda esquerda, 30px da borda superior.
    cv2.putText(
        frame,
        texto_contador,
        (10, 30),                    # posição fixa no canto superior esquerdo
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,                         # tamanho da fonte levemente maior para leitura fácil
        (255, 255, 255),             # cor branca para contrastar com qualquer fundo
        2
    )

    # --- ETAPA 6: LÓGICA BÁSICA DE ESTADO DE FADIGA ---

    # Se nenhum olho foi detectado no rosto (0 olhos visíveis),
    # pode indicar que o trabalhador está com os olhos fechados.
    # Exibimos um alerta visual em VERMELHO na tela.
    # ATENÇÃO: Esta é uma lógica INICIAL e simples. Um olho fora do ângulo
    # da câmera também pode resultar em 0 detecções — refinamos isso nas próximas fases.
    if len(rostos_detectados) > 0 and total_olhos_frame == 0:
        cv2.putText(
            frame,
            "ATENCAO: OLHOS NAO DETECTADOS",
            (10, 65),                # posição abaixo do contador de olhos
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),             # BGR: Vermelho
            2
        )

    # Exibe o frame final com todos os desenhos
    cv2.imshow("MATI - Monitoramento de Rosto", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Encerrando o MATI...")
        break

# --- ENCERRAMENTO SEGURO ---

cap.release()
cv2.destroyAllWindows()
print("Câmera liberada. Sistema encerrado com segurança.")
