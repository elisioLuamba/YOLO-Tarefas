# 🚀 Configuração do Ambiente Conda para YOLO

Estrutura organizada para projetos com:

- Ultralytics YOLO
- OpenCV
- PyTorch
- CUDA
- Visão Computacional
- IA em Tempo Real

---

# 📦 Criando o Ambiente Conda

Crie um ambiente isolado para evitar conflitos de dependências.

```bash
conda create -n YOLO-Tarefas python=3.10 -y
```

---

# 🧠 Explicando o Comando

## `-n`

Define o nome do ambiente.

```bash
conda create -n YOLO-Tarefas
```

Resultado:

```text
YOLO-Tarefas
```

---

## `python=3.10`

Define a versão do Python usada no ambiente.

O Python 3.10 é amplamente utilizado em IA e Visão Computacional por oferecer:

- alta compatibilidade
- estabilidade
- suporte completo ao PyTorch
- suporte maduro ao CUDA
- menos conflitos de dependências

---

## `-y`

Significa:

```text
--yes
```

Aceita automaticamente as confirmações da instalação.

Sem `-y`:

```text
Proceed ([y]/n)?
```

Com `-y`:

```text
Instala automaticamente
```

---

# ⚡ Ativando o Ambiente

```bash
conda activate YOLO-Tarefas
```

Quando estiver ativo:

```text
(YOLO-Tarefas)
```

---

# 🖥️ Instalação Básica (CPU)

Ideal para:

- estudos
- testes
- notebooks
- computadores sem GPU NVIDIA

```bash
pip install ultralytics opencv-python
```

---

# 🔥 Instalação com GPU NVIDIA (CUDA)

Para processamento acelerado por hardware.

---

## 1️⃣ Ativar Ambiente

```bash
conda activate YOLO-Tarefas
```

---

## 2️⃣ Instalar PyTorch com CUDA

Exemplo usando CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3️⃣ Instalar YOLO + OpenCV

```bash
pip install ultralytics opencv-python
```

---

# 🧪 Testando a Instalação

```python
import torch
import ultralytics

print("Versão Ultralytics:", ultralytics.__version__)
print("CUDA disponível?:", torch.cuda.is_available())
```

---

# ✅ Resultado Esperado

```text
CUDA disponível?: True
```

Se retornar:

```text
False
```

o YOLO está utilizando CPU.

---

# 🤖 Como Funciona a Versão do YOLO

O pacote principal é:

```bash
pip install ultralytics
```

A versão do YOLO é escolhida no código:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

ou:

```python
model = YOLO("yolo11n.pt")
```

---

# 📚 Modelos YOLO

| Modelo | Uso |
|---|---|
| yolov8n.pt | Nano / Muito leve |
| yolov8s.pt | Small |
| yolov8m.pt | Medium |
| yolov8l.pt | Large |
| yolov8x.pt | Extra Large |

---

# 🧠 Significado das Letras

| Letra | Significado |
|---|---|
| n | nano |
| s | small |
| m | medium |
| l | large |
| x | extra large |

Quanto maior o modelo:

- maior precisão
- maior consumo de VRAM
- mais processamento
- menor FPS em hardware fraco

A física continua vencendo discussões na internet. Infelizmente.

---

# 📂 Estrutura Recomendada do Projeto

```text
YOLO-Tarefas/
│
├── datasets/
├── models/
├── outputs/
├── videos/
├── images/
├── scripts/
│
├── main.py
├── train.py
├── detect.py
└── requirements.txt
```

---

# 🧰 Bibliotecas Recomendadas

| Biblioteca | Função |
|---|---|
| OpenCV | Manipulação de imagem/vídeo |
| PyTorch | Deep Learning |
| MediaPipe | Rastreamento corporal |
| NumPy | Matrizes |
| supervision | Visualização |
| cvzone | Utilidades CV |

---

# ⚡ Combinações Poderosas

## YOLO + MediaPipe

- Pose Estimation
- Rastreamento corporal
- Ergonomia
- Monitoramento de fadiga

---

## YOLO + Arduino

- Automação industrial
- Robótica
- Sensores inteligentes
- Sistemas embarcados
- Esteiras inteligentes

---

# 🛠️ Comandos Úteis do Conda

## Listar Ambientes

```bash
conda env list
```

---

## Remover Ambiente

```bash
conda remove -n YOLO-Tarefas --all
```

---

## Atualizar Conda

```bash
conda update conda
```

---

# 🎮 Verificar GPU NVIDIA

```bash
nvidia-smi
```

Se funcionar corretamente, os drivers NVIDIA estão instalados.

Se não funcionar... então começa a clássica jornada espiritual chamada "configuração de CUDA". Uma experiência que une sofrimento, fóruns de 2019 e decisões questionáveis da humanidade.

---

# 🚀 Stack Recomendada

```text
Python 3.10
Ultralytics YOLO
PyTorch
OpenCV
CUDA
MediaPipe
```

---

# 🧠 Áreas de Aplicação

- Visão Computacional
- IA em Tempo Real
- Robótica
- Segurança Inteligente
- Ergonomia Industrial
- Monitoramento Corporal
- Detecção de Objetos
- Pose Estimation
- Automação com Arduino
- Sistemas Inteligentes

### Code
```python
import cv2
from ultralytics import YOLO

# Carregar o modelo pré-treinado YOLOv8n
model = YOLO('yolov8n.pt')

# Inicializa a webcam (0 para a webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, stream=True)
    
    for r in results:
        frame = r.plot()
        
    
    cv2.imshow("Tarefas", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey()
```
