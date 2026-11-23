import cv2
import time
import numpy as np
import winsound  # Para sons no Windows

# Configurações de som
BEEP_FREQUENCY = 2000  # Frequência do beep em Hertz
BEEP_DURATION = 500    # Duração do beep em milissegundos

# Classificadores Haar para rosto e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Caminhos para os arquivos YOLO
model_cfg = "models/yolov3.cfg"
model_weights = "models/yolov3.weights"
coco_names = "models/coco.names"

# Carrega as classes que o YOLO pode detectar
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carrega o modelo YOLO
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Utiliza CUDA
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Parâmetros de detecção de fadiga ocular
EYE_CLOSED_THRESHOLD = 1.5  # Tempo limite com os olhos fechados (segundos)
EYE_DETECTION_THRESHOLD = 5  # Frames consecutivos sem detectar olhos
eye_closed_time = 0
no_eye_frames = 0
last_time = time.time()

# Captura de vídeo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Função de alerta
def emitir_alerta(tipo):
    if tipo == "celular":
        print("Alerta: Uso do celular detectado!")
        winsound.Beep(1000, 500)
    elif tipo == "fadiga":
        print("Alerta: Você está com os olhos fechados! Acorde!")
        winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)

# Função para aplicar supressão não máxima (NMS)
def aplicar_nms(boxes, confidences, threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
    return indices

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona a imagem para acelerar o processamento
    height, width, channels = frame.shape
    small_frame = cv2.resize(frame, (416, 416))

    # YOLO: Pré-processamento e detecção
    blob = cv2.dnn.blobFromImage(small_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(net.getUnconnectedOutLayersNames())

    celular_detectado = False
    boxes = []
    confidences = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id < len(classes) and classes[class_id] == "cell phone" and confidence > 0.5:
                center_x, center_y, w, h = (
                    int(obj[0] * width),
                    int(obj[1] * height),
                    int(obj[2] * width),
                    int(obj[3] * height),
                )
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    if boxes:
        indices = aplicar_nms(boxes, confidences)
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Celular", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            celular_detectado = True

    if celular_detectado:
        emitir_alerta("celular")
        cv2.putText(frame, "ATENCAO: VOCE ESTA NO CELULAR!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Fadiga ocular: Rosto e olhos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    state_text = "Olhos abertos"
    state_color = (0, 255, 0)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        if len(eyes) == 0:
            no_eye_frames += 1
        else:
            no_eye_frames = 0

        if no_eye_frames >= EYE_DETECTION_THRESHOLD:
            eye_closed_time += time.time() - last_time
            state_text = "Olhos fechados"
            state_color = (0, 0, 255)
        else:
            eye_closed_time = 0

        last_time = time.time()

        if eye_closed_time >= EYE_CLOSED_THRESHOLD:
            emitir_alerta("fadiga")
            state_text = "ATENCAO: VOCE ESTA DORMINDO!"
            eye_closed_time = 0

    cv2.putText(frame, state_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
    cv2.imshow("Deteccao Integrada", frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

