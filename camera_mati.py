# ============================================================
# PROJETO MATI — Monitoramento Avançado do Trabalhador Inteligente
# Arquivo: camera_mati.py
# Objetivo: Ligar a webcam e exibir o vídeo em tempo real
# ============================================================

# Importa a biblioteca OpenCV.
# Apesar de instalarmos como "opencv-python", ela é chamada de "cv2" no código.
# O OpenCV (Open Source Computer Vision) é a principal biblioteca
# do mundo para trabalhar com imagens e vídeos em tempo real.
import cv2

# --- ETAPA 1: CONECTAR À WEBCAM ---

# cv2.VideoCapture(0) abre a câmera conectada ao computador.
# O número 0 significa "primeira câmera disponível" (sua webcam principal).
# Se você tivesse duas câmeras, usaria VideoCapture(1) para a segunda.
# O resultado é um objeto "câmera" guardado na variável "cap" (de capture).
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta com sucesso.
# cap.isOpened() retorna True se a câmera está funcionando, ou False se houve erro.
if not cap.isOpened():
    # Se a câmera não abriu, imprime uma mensagem de erro e encerra o script.
    # Isso é uma boa prática: sempre verifique se o recurso foi iniciado corretamente.
    print("ERRO: Não foi possível acessar a webcam.")
    # exit() encerra o programa imediatamente.
    exit()

# Confirma no terminal que a câmera foi ligada com sucesso.
print("Câmera iniciada com sucesso! Pressione 'q' para encerrar.")

# --- ETAPA 2: LOOP PRINCIPAL DE LEITURA DE FRAMES ---

# while True cria um loop infinito.
# O programa ficará aqui dentro, lendo e exibindo frames,
# até que uma condição de saída seja ativada (a tecla 'q').
while True:

    # cap.read() captura um único frame (foto instantânea) da câmera.
    # Ele retorna DOIS valores ao mesmo tempo:
    #   - "ret" (return): um booleano True/False indicando se a leitura foi bem-sucedida.
    #   - "frame": a imagem em si, representada como uma matriz de pixels (array numpy).
    ret, frame = cap.read()

    # Verifica se o frame foi capturado corretamente.
    # Se ret for False, significa que algo deu errado (ex: câmera desconectada).
    if not ret:
        print("ERRO: Falha ao capturar o frame da câmera.")
        # "break" interrompe o loop while e vai para o código de encerramento.
        break

    # --- ETAPA 3: EXIBIR O FRAME NA TELA ---

    # cv2.imshow() abre uma janela e exibe a imagem.
    # Primeiro argumento: o título da janela (string).
    # Segundo argumento: a imagem/frame a ser exibido.
    cv2.imshow("MATI - Monitoramento de Rosto", frame)

    # --- ETAPA 4: DETECTAR O PRESSIONAMENTO DA TECLA 'q' ---

    # cv2.waitKey(1) pausa a execução por 1 milissegundo e
    # aguarda o pressionamento de uma tecla no teclado.
    # O valor 1 é importante: sem essa pausa, a janela não consegue
    # processar eventos e travaria na primeira imagem.
    #
    # Ele retorna o código ASCII da tecla pressionada.
    # ord('q') retorna o código ASCII da letra 'q', que é 113.
    # O & 0xFF é uma máscara de bits necessária em alguns sistemas
    # operacionais (especialmente Linux/Mac) para garantir compatibilidade.
    # Em Windows geralmente funciona sem ela, mas é boa prática sempre incluir.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Encerrando o MATI...")
        # "break" sai do loop while e vai para o bloco de encerramento abaixo.
        break

# --- ETAPA 5: ENCERRAMENTO E LIBERAÇÃO DE RECURSOS ---
# Este bloco é executado assim que o loop é interrompido pelo "break".

# cap.release() libera a webcam, devolvendo o controle dela ao sistema operacional.
# SEMPRE faça isso ao terminar — sem isso, a câmera pode ficar "presa"
# e outros programas não conseguirão usá-la.
cap.release()

# cv2.destroyAllWindows() fecha todas as janelas de imagem abertas pelo OpenCV.
# Sem este comando, a janela pode ficar "fantasma" na tela.
cv2.destroyAllWindows()

print("Câmera liberada. Sistema encerrado com segurança.")