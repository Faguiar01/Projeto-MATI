import cv2

# 1. Carrega o classificador (o mapa de rostos que você baixou)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Liga a webcam
cap = cv2.VideoCapture(0)

print("MATI Iniciado. Pressione 'q' para encerrar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # O computador detecta melhor em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Executa a detecção (ajusta escala e vizinhos para precisão)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 4. Desenha o feedback visual para cada rosto encontrado
    for (x, y, w, h) in faces:
        # Retângulo Verde (BGR: 0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "TRABALHADOR ATIVO", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Exibe o resultado final
    cv2.imshow('MATI - Inteligencia Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()