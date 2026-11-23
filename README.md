# Alert Driver

Sistema de monitoramento em tempo real para prevenção de acidentes rodoviários causados por **sonolência** e **distração pelo uso de celular** ao volante, desenvolvido como projeto acadêmico no 1º semestre de Engenharia de Controle e Automação (UFSC).

## Sobre o Projeto

O **Alert Driver** combina visão computacional em Python (OpenCV + YOLOv3 + Haar Cascades) com um protótipo de aplicativo mobile desenvolvido no Kodular para acompanhar o motorista, emitir alertas sonoros imediatos em situações de risco e apresentar métricas de condução.

### Funcionalidades
- **Detecção de sonolência (`src/detector_sonolencia.py`):** identifica rosto e olhos via Haar Cascades (`haarcascade_frontalface_default.xml` e `haarcascade_eye.xml`) e dispara alerta sonoro quando os olhos permanecem fechados além do limite configurado.
- **Detecção de uso de celular (`src/detector_celular.py`):** utiliza YOLOv3 (`models/yolov3.cfg` + `models/coco.names`) com supressão não máxima (NMS) para identificar o uso de celular na cabine e emitir aviso visual e sonoro.
- **Sistema integrado (`src/alert_driver.py`):** executa simultaneamente a detecção de fadiga ocular e de uso de celular em tempo real.
- **Protótipo do aplicativo (`app/AlertDriverV3.aia`):** projeto Kodular com telas de entrada/login, painel inicial com indicadores de cansaço, pausas e horas dirigindo, histórico por data, mapa e perfil.

## Estrutura do Repositório

```text
alert-driver/
├── app/
│   └── AlertDriverV3.aia        # Projeto Kodular sanitizado (sem chaves de API)
├── models/
│   ├── coco.names               # Classes do dataset COCO para o YOLOv3
│   └── yolov3.cfg               # Configuração da rede YOLOv3
├── src/
│   ├── alert_driver.py          # Script principal integrado (sonolência + celular)
│   ├── detector_celular.py      # Detector isolado de uso de celular
│   └── detector_sonolencia.py   # Detector isolado de fadiga ocular
├── .gitignore
├── requirements.txt
└── README.md
```

## Como Executar

### 1. Instalar dependências
> **Nota:** os scripts utilizam `winsound` para alertas sonoros e `cv2.CAP_DSHOW` no script integrado, sendo voltados originalmente para ambiente Windows com webcam conectada.

```bash
pip install -r requirements.txt
```

### 2. Baixar os pesos do YOLOv3
O arquivo `yolov3.weights` (~237 MB) excede o limite de tamanho do GitHub e não está versionado neste repositório. Baixe-o e salve em `models/yolov3.weights`:

```bash
curl -L -o models/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
```

### 3. Executar
A partir da raiz do repositório:

```bash
# Sistema integrado (fadiga + celular)
python src/alert_driver.py

# Apenas detecção de sonolência
python src/detector_sonolencia.py

# Apenas detecção de celular
python src/detector_celular.py
```

Pressione `q` para encerrar a captura.

### 4. Abrir o aplicativo no Kodular
Importe `app/AlertDriverV3.aia` no [Kodular Creator](https://creator.kodular.io/). Caso utilize a tela de mapa (`Mapa`), configure sua própria chave do Google Maps nas propriedades do projeto antes de compilar.

## Equipe
- **Luiz Gustavo de Andrade dos Santos**
- **Luiz Antonio Zilli**
- **Manuella Ariotti**
