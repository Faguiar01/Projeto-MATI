# =============================================================================
# PROJETO MATI - Script de Teste: Detecção de Emoção em Tempo Real
# Autor: Engenharia de Software / Visão Computacional
# Conformidade LGPD: Nenhuma imagem é salva em disco. Apenas o TEXTO da emoção
# é extraído e utilizado.
# =============================================================================

# --- IMPORTAÇÕES ---

# 'cv2' é o OpenCV. Usamos ele para acessar a câmera e desenhar na tela.
import cv2

# 'DeepFace' é a classe principal da biblioteca deepface.
# Ela encapsula modelos de IA prontos para análise facial (emoção, idade, etc.).
from deepface import DeepFace

# 'time' é uma biblioteca nativa do Python para trabalhar com tempo.
# Vamos usá-la para controlar o intervalo entre cada análise da IA.
import time


# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================

# Intervalo em segundos entre cada chamada ao DeepFace.
# Valor 1.5 significa: a IA analisa o rosto a cada 1.5 segundos.
# ESTE É O "PULO DO GATO" de performance. Se chamarmos a IA em TODO frame
# (~30x por segundo), o vídeo congela. Chamando a cada 1.5s, o vídeo fica fluido.
INTERVALO_ANALISE_SEGUNDOS = 1.5

# Variável que vai guardar o texto da última emoção detectada.
# Começa como "Analisando..." para ter algo para mostrar na tela enquanto
# a primeira análise ainda não terminou.
emocao_atual = "Analisando..."

# Variável que guarda o MOMENTO em que fizemos a última análise.
# 'time.time()' retorna o tempo atual em segundos (ex: 1719432000.123).
# Ao subtrair duas dessas medições, obtemos quantos segundos se passaram.
ultimo_tempo_analise = time.time()


# =============================================================================
# INICIALIZAÇÃO DA CÂMERA
# =============================================================================

# 'cv2.VideoCapture(0)' abre a câmera de índice 0, que é geralmente a webcam
# padrão do seu computador. Se você tiver mais câmeras, tente 1, 2, etc.
camera = cv2.VideoCapture(0)

# Verificamos se a câmera foi aberta com sucesso.
# Se não foi (ex: permissão negada, câmera ocupada), encerramos o script
# com uma mensagem de erro clara.
if not camera.isOpened():
    print("[ERRO] Não foi possível acessar a webcam. Verifique as permissões.")
    exit()  # Encerra o programa imediatamente.

print("[INFO] Câmera iniciada. Pressione 'Q' para sair.")
print("[LGPD] Nenhuma imagem será salva em disco. Apenas o texto da emoção é processado.")


# =============================================================================
# LOOP PRINCIPAL DE CAPTURA E ANÁLISE
# =============================================================================

# 'while True' cria um loop infinito. O vídeo ficará rodando até que
# o usuário pressione 'Q' para sair (tratamos isso no final do loop).
while True:

    # --- LEITURA DO FRAME ---

    # 'camera.read()' captura UM frame (uma foto) da câmera.
    # Retorna dois valores:
    #   'capturou_frame': True se a leitura foi bem-sucedida, False se falhou.
    #   'frame': o array de pixels da imagem (um array NumPy com formato BGR).
    capturou_frame, frame = camera.read()

    # Se por algum motivo não conseguimos capturar o frame (ex: câmera
    # desconectada durante o uso), saímos do loop.
    if not capturou_frame:
        print("[ERRO] Falha ao capturar frame da câmera. Encerrando.")
        break  # Sai do 'while True'.

    # --- VERIFICAÇÃO DE TEMPO PARA ANÁLISE DA IA ---

    # 'time.time()' pega o tempo AGORA.
    # Subtraímos o 'ultimo_tempo_analise' para saber quantos segundos
    # se passaram desde a última vez que chamamos o DeepFace.
    tempo_agora = time.time()
    tempo_decorrido = tempo_agora - ultimo_tempo_analise

    # Só executamos a análise pesada da IA se o intervalo configurado passou.
    if tempo_decorrido >= INTERVALO_ANALISE_SEGUNDOS:

        # --- ANÁLISE FACIAL COM DEEPFACE ---

        # Usamos um bloco try/except porque o DeepFace pode lançar um erro
        # se não encontrar nenhum rosto no frame. Sem o try/except,
        # esse erro quebraria o programa inteiro. Com ele, simplesmente
        # ignoramos o frame sem rosto e continuamos.
        try:
            # 'DeepFace.analyze()' é o coração do script.
            # Argumentos:
            #   img_path=frame   -> passamos o frame do OpenCV DIRETAMENTE (array NumPy).
            #                       NENHUM arquivo é criado. Isso garante conformidade LGPD.
            #   actions=['emotion']  -> pedimos APENAS a análise de emoção.
            #                          Poderíamos pedir 'age', 'gender', 'race', mas
            #                          não precisamos e isso seria desnecessário (princípio
            #                          da minimização de dados da LGPD).
            #   enforce_detection=False -> Se não achar rosto, retorna resultado genérico
            #                             em vez de lançar erro. Torna o sistema mais robusto.
            resultado = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False
            )

            # 'resultado' é uma LISTA de dicionários, um para cada rosto detectado.
            # Pegamos o primeiro rosto detectado: resultado[0].
            # Dentro dele, a chave 'dominant_emotion' guarda a emoção mais provável.
            # Exemplos de valor: 'happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust'
            emocao_atual = resultado[0]['dominant_emotion']

            # Imprime a emoção no terminal. Útil para debug e para confirmar
            # que o sistema está funcionando. Em produção, este dado poderia
            # ser enviado para um banco de dados ou API — nunca o frame/imagem.
            print(f"[EMOÇÃO DETECTADA] {emocao_atual.upper()}")

        except Exception as erro:
            # Se o DeepFace lançar qualquer erro (rosto não encontrado,
            # problema no modelo), atualizamos a emoção para indicar isso
            # e continuamos o loop sem travar.
            emocao_atual = "Rosto nao detectado"
            # 'pass' seria suficiente para ignorar, mas imprimir ajuda no debug.
            # print(f"[AVISO] {erro}")  # Descomente esta linha se quiser ver o erro.

        # Atualizamos o 'ultimo_tempo_analise' para o momento atual.
        # Isso reinicia o contador para a próxima análise.
        ultimo_tempo_analise = time.time()

    # --- DESENHO VISUAL NA TELA (OVERLAY) ---

    # Mesmo quando a IA NÃO está analisando, sempre desenhamos o overlay.
    # Isso garante que o vídeo seja fluido: o frame é atualizado ~30x/segundo,
    # mas a chamada pesada da IA ocorre apenas a cada 1.5 segundos.

    # Para encontrar o rosto e desenhar o retângulo, usamos o classificador
    # Haar Cascade do OpenCV — ele é MUITO mais leve que o DeepFace para
    # essa tarefa simples de detecção de posição.

    # Carregamos o modelo de detecção de faces (já vem instalado com o OpenCV).
    detector_rosto = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Convertemos o frame para escala de cinza.
    # O detector Haar Cascade trabalha em escala de cinza — é mais eficiente.
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 'detectMultiScale' procura por rostos no frame cinza.
    # Retorna uma lista de retângulos (x, y, largura, altura) para cada rosto.
    #   scaleFactor=1.1   -> quão rápido o tamanho da janela de busca é reduzido.
    #   minNeighbors=5    -> quantos vizinhos um retângulo precisa ter para ser válido.
    #                        Valores maiores = menos falsos positivos.
    #   minSize=(80, 80)  -> tamanho mínimo do rosto para ser detectado (em pixels).
    rostos = detector_rosto.detectMultiScale(
        frame_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Iteramos sobre cada rosto encontrado.
    # 'x, y' são as coordenadas do canto superior esquerdo do rosto.
    # 'w, h' são a largura (width) e altura (height) do rosto.
    for (x, y, w, h) in rostos:

        # Desenhamos um retângulo ao redor do rosto no frame ORIGINAL (colorido).
        # cv2.rectangle(imagem, ponto_inicio, ponto_fim, cor_BGR, espessura)
        # Cor (0, 255, 0) = Verde no formato BGR do OpenCV.
        # Espessura 2 = linha com 2 pixels de largura.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Escrevemos o texto da emoção ACIMA do retângulo do rosto.
        # 'y - 10' coloca o texto 10 pixels acima da borda superior do retângulo.
        # Se y for muito pequeno (rosto no topo da tela), usamos max() para
        # garantir que o texto não saia da tela (mínimo y=20).
        posicao_texto = (x, max(y - 10, 20))

        # cv2.putText(imagem, texto, posicao, fonte, escala, cor, espessura)
        cv2.putText(
            frame,                         # Imagem onde escrever
            emocao_atual.upper(),          # Texto: emoção em maiúsculas
            posicao_texto,                 # Posição (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,      # Fonte: simples e legível
            0.9,                           # Escala da fonte (tamanho)
            (0, 255, 0),                   # Cor: verde (BGR)
            2                              # Espessura do texto
        )

    # --- EXIBIÇÃO DO FRAME ---

    # 'cv2.imshow()' abre uma janela e exibe o frame atual com todos os desenhos.
    # O primeiro argumento é o título da janela.
    cv2.imshow("MATI - Detector de Emocao (Pressione Q para sair)", frame)

    # --- CAPTURA DO COMANDO DE SAÍDA ---

    # 'cv2.waitKey(1)' aguarda 1 milissegundo por um evento de teclado.
    # Retorna o código ASCII da tecla pressionada.
    # '& 0xFF' é uma máscara de bits para garantir compatibilidade entre sistemas.
    # 'ord('q')' retorna o código ASCII da letra 'q' (minúscula).
    # Se o usuário pressionar 'Q' ou 'q', saímos do loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Tecla 'Q' pressionada. Encerrando o script.")
        break  # Sai do 'while True'.


# =============================================================================
# LIMPEZA DE RECURSOS (sempre executado ao sair do loop)
# =============================================================================

# Libera o recurso da câmera para que outros programas possam usá-la.
# Boa prática essencial — sem isso, a câmera pode ficar "presa".
camera.release()

# Fecha todas as janelas abertas pelo OpenCV.
cv2.destroyAllWindows()

print("[INFO] Recursos liberados. Script encerrado com sucesso.")