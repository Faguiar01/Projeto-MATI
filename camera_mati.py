# ============================================================
# PROJETO MATI — Monitoramento Avançado do Trabalhador Inteligente
# Arquivo: camera_mati.py
# Versão 4.0 — Detecção de Fadiga + Registro Automático em CSV
# ============================================================

# --- IMPORTAÇÕES ---

import cv2

# "csv" é uma biblioteca NATIVA do Python (não precisa de pip install).
# Ela oferece ferramentas para ler e ESCREVER arquivos .csv de forma segura,
# lidando automaticamente com vírgulas dentro de texto, aspas, etc.
import csv

# "datetime" também é nativa do Python.
# O módulo "datetime" dentro da biblioteca "datetime" (sim, mesmo nome)
# permite capturar e formatar data e hora do sistema operacional.
from datetime import datetime

# --- ETAPA 1: CARREGAR CLASSIFICADORES ---

classificador_rosto = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificador_olho  = cv2.CascadeClassifier("haarcascade_eye.xml")

if classificador_rosto.empty():
    print("ERRO: haarcascade_frontalface_default.xml não encontrado.")
    exit()

if classificador_olho.empty():
    print("ERRO: haarcascade_eye.xml não encontrado.")
    exit()

print("Classificadores carregados com sucesso!")

# --- ETAPA 2: CONFIGURAÇÕES DO LOG ---

# Nome do arquivo CSV onde os eventos serão registrados.
ARQUIVO_CSV = "dados_mati.csv"

# BPM fixo por enquanto (será substituído por leitura real futuramente).
BPM_FIXO = 80

# Peças produzidas: mantemos o último valor registrado.
# Começa em 0 e será atualizado conforme o sistema evoluir.
pecas_produzidas = 0

# --- [NOVO] VARIÁVEIS DE CONTROLE DA TRAVA ANTI-FLOOD ---
#
# O problema sem trava: o loop roda ~30 vezes por segundo.
# Sem controle, um alerta de 10 segundos gravaria ~300 linhas no CSV.
# A solução: guardamos o MOMENTO em que salvamos pela última vez,
# e só permitimos um novo save se já passaram 5 segundos desde o anterior.
#
# datetime.min é o menor valor possível de data/hora no Python.
# Usamos como valor inicial para garantir que o PRIMEIRO alerta
# sempre seja salvo imediatamente (qualquer hora atual será maior que datetime.min).
ultimo_registro = datetime.min

# Intervalo mínimo entre registros: 5 segundos.
# timedelta representa uma "duração de tempo" — aqui, 5 segundos.
from datetime import timedelta
INTERVALO_MINIMO = timedelta(seconds=5)

# --- ETAPA 3: FUNÇÃO PARA SALVAR NO CSV ---

def salvar_evento_fadiga(horario, bpm, status, pecas):
    # SOBRE O MODO 'a' (APPEND — ACRESCENTAR):
    #
    # Quando abrimos um arquivo com open(), o segundo argumento define o MODO:
    #
    #   'w' (write)  → APAGA tudo e começa do zero. PERIGOSO para nosso caso.
    #   'r' (read)   → Apenas leitura. Não permite escrever nada.
    #   'a' (append) → Abre o arquivo e posiciona o cursor NO FINAL.
    #                  Tudo que escrevemos é ADICIONADO após o conteúdo existente.
    #                  Se o arquivo não existir, ele é CRIADO automaticamente.
    #
    # Para o MATI, 'a' é perfeito: cada evento de fadiga vira uma nova linha
    # sem jamais apagar os registros anteriores do dia.
    #
    # newline='' é necessário no Windows para o csv.writer não inserir
    # linhas em branco extras entre cada registro gravado.
    # encoding='utf-8' garante que acentos e caracteres especiais sejam salvos corretamente.
    with open(ARQUIVO_CSV, mode='a', newline='', encoding='utf-8') as arquivo:

        # csv.writer() cria um "escritor" que formata os dados automaticamente
        # no padrão CSV: separa com vírgulas, adiciona aspas quando necessário, etc.
        escritor = csv.writer(arquivo)

        # escritor.writerow() escreve UMA LINHA no arquivo.
        # Passamos uma lista Python: cada item vira uma coluna no CSV.
        # A ordem deve ser a mesma do cabeçalho: Horario, BPM, Status_Rosto, Pecas_Produzidas
        escritor.writerow([horario, bpm, status, pecas])

    # Confirma no terminal que o registro foi feito (útil para debug)
    print(f"[LOG] Evento salvo: {horario} | {status}")

# --- ETAPA 4: CONECTAR À WEBCAM ---

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERRO: Não foi possível acessar a webcam.")
    exit()

print("Câmera iniciada! Pressione 'q' para encerrar.")

# --- ETAPA 5: LOOP PRINCIPAL ---

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERRO: Falha ao capturar o frame.")
        break

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostos_detectados = classificador_rosto.detectMultiScale(
        frame_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    total_olhos_frame = 0

    for (x, y, w, h) in rostos_detectados:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray  = frame_cinza[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        olhos_detectados = classificador_olho.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(20, 20)
        )

        total_olhos_frame += len(olhos_detectados)

        for (ex, ey, ew, eh) in olhos_detectados:
            centro_x = ex + ew // 2
            centro_y = ey + eh // 2
            raio     = min(ew, eh) // 2
            cv2.circle(roi_color, (centro_x, centro_y), raio, (255, 0, 0), 2)

    # --- EXIBIR CONTADOR DE OLHOS ---

    cv2.putText(
        frame,
        f"Olhos Detectados: {total_olhos_frame}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --- [NOVO] LÓGICA DE ALERTA + GRAVAÇÃO COM TRAVA ANTI-FLOOD ---

    # Condição de fadiga: rosto visível MAS olhos não detectados
    if len(rostos_detectados) > 0 and total_olhos_frame == 0:

        # Exibe o alerta visual vermelho na tela (igual à versão anterior)
        cv2.putText(
            frame,
            "ATENCAO: OLHOS NAO DETECTADOS",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # --- VERIFICAÇÃO DA TRAVA DE TEMPO ---
        #
        # datetime.now() retorna o momento EXATO atual (data + hora + microssegundos).
        # Ex: 2025-06-10 14:35:22.847291
        agora = datetime.now()

        # Calculamos quanto tempo passou desde o último registro.
        # Subtrair dois objetos datetime resulta em um timedelta (duração).
        # Se essa duração for MAIOR que 5 segundos (INTERVALO_MINIMO),
        # liberamos um novo registro. Caso contrário, ignoramos este frame.
        tempo_desde_ultimo = agora - ultimo_registro

        if tempo_desde_ultimo > INTERVALO_MINIMO:

            # Formata o horário como string legível para salvar no CSV.
            # strftime = "string format time"
            # %Y = ano com 4 dígitos | %m = mês | %d = dia
            # %H = hora (24h)        | %M = minuto | %S = segundo
            horario_formatado = agora.strftime("%Y-%m-%d %H:%M:%S")

            # Chama a função que grava a linha no CSV
            salvar_evento_fadiga(
                horario=horario_formatado,
                bpm=BPM_FIXO,
                status="Fadiga Detectada",
                pecas=pecas_produzidas
            )

            # Atualiza o marcador de tempo para o momento atual.
            # Isso "reinicia o cronômetro" da trava.
            ultimo_registro = agora

    # Exibe o frame com todos os elementos desenhados
    cv2.imshow("MATI - Monitoramento de Rosto", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Encerrando o MATI...")
        break

# --- ENCERRAMENTO SEGURO ---

cap.release()
cv2.destroyAllWindows()
print("Câmera liberada. Sistema encerrado com segurança.")