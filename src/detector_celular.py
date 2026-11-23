import cv2
import numpy as np
import winsound  # Biblioteca para sons no Windows

# Caminhos para os arquivos YOLO
model_cfg = "models/yolov3.cfg"
model_weights = "models/yolov3.weights"
coco_names = "models/coco.names"

# Carrega as classes que o YOLO pode detectar
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carrega o modelo YOLO
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Define a função de alertaq
def emitir_alerta():
    print("Alerta: Uso do celular detectado!")
    # Emite um som usando winsound
    winsound.Beep(1000, 500)  # Frequência de 1000 Hz por 500 ms

# Captura de vídeo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Ajuste a resolução da câmera (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # Largura
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Altura

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break

    # Aplicação de suavização para redução de ruído
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    height, width, channels = frame.shape

    # Pré-processamento da imagem para o YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Listas para armazenar os resultados de NMS
    class_ids = []
    confidences = []
    boxes = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id < len(classes) and classes[class_id] == "cell phone" and confidence > 0.5:
                # Coordenadas do objeto detectado
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Armazenar detecção
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Aplica o NMS (supressão não máxima) para evitar múltiplos retângulos para o mesmo objeto
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    celular_detectado = False

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Celular", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            celular_detectado = True

    # Emite o alerta se o celular for detectado
    if celular_detectado:
        emitir_alerta()
        # Adiciona a mensagem na tela
        cv2.putText(
            frame,
            "ATENCAO VOCE ESTA NO CELULAR!",
            (50, 50),  # Posição na tela
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Tamanho da fonte
            (0, 0, 255),  # Cor do texto (vermelho)
            2,  # Espessura
            cv2.LINE_AA,
        )

    # Exibir o frame
    cv2.imshow("Deteccao de Celular", frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()