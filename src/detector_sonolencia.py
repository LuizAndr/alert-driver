import cv2
import time
import winsound

# Configurações de som para o beep
BEEP_FREQUENCY = 2000  # Frequência do beep em Hertz
BEEP_DURATION = 500  # Duração do beep em milissegundos

# Classificadores Haar para rosto e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Verifica se os classificadores foram carregados corretamente
if face_cascade.empty() or eye_cascade.empty():
    print("Erro ao carregar os classificadores Haar.")
    exit()

# Parâmetros de detecção de fadiga
EYE_CLOSED_THRESHOLD = 1.0  # Tempo limite com os olhos fechados (segundos)
EYE_DETECTION_THRESHOLD = 5  # Número de frames consecutivos sem detectar olhos
eye_closed_time = 0  # Tempo acumulado com os olhos fechados
last_time = time.time()  # Marca o tempo da última verificação
no_eye_frames = 0  # Contador de frames consecutivos sem olhos detectados

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Melhorar imagem com escala de cinza e equalização de histograma
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Equalização do histograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    state_text = "Olhos abertos"  # Texto padrão para o estado dos olhos
    state_color = (0, 255, 0)  # Verde (BGR) para olhos abertos

    for (x, y, w, h) in faces:
        # Desenha o retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define a região de interesse (ROI) para buscar olhos
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detecta olhos na ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        if len(eyes) == 0:
            # Incrementa o contador de frames sem olhos detectados
            no_eye_frames += 1
        else:
            # Reseta o contador se olhos forem detectados
            no_eye_frames = 0

        # Atualiza o tempo de olhos fechados se não detectar olhos consecutivamente
        if no_eye_frames >= EYE_DETECTION_THRESHOLD:
            eye_closed_time += time.time() - last_time
            state_text = "Olhos fechados"
            state_color = (0, 0, 255)  # Vermelho
        else:
            eye_closed_time = 0

        last_time = time.time()  # Atualiza o tempo

        # Se os olhos ficarem fechados por mais que o limite, dispara o alarme
        if eye_closed_time >= EYE_CLOSED_THRESHOLD:
            print("Atenção: Você está com os olhos fechados! Acorde!")
            cv2.putText(
                frame,
                "ATENCAO: VOCE ESTA DORMINDO!",
                (50, 100),  # Posição (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,  # Fonte do texto
                1,  # Tamanho da fonte
                (0, 0, 255),  # Cor do texto (vermelho)
                2,  # Espessura da linha do texto
                cv2.LINE_AA  # Tipo de linha
            )
            # Emite o som de alerta usando Beep
            winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)
            time.sleep(1)  # Pausa para evitar tocar o som repetidamente
            eye_closed_time = 0  # Reseta o contador

    # Exibe o estado dos olhos na tela
    cv2.putText(
        frame,
        state_text,
        (50, 50),  # Posição do texto
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Tamanho da fonte
        state_color,  # Cor do texto
        2,  # Espessura da linha do texto
        cv2.LINE_AA
    )

    cv2.imshow('Detecção de Fadiga', frame)

    # Pressiona 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
