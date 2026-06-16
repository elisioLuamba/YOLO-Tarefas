import cv2
import numpy as np
import random
from ultralytics import YOLO

# 1. MODELO MAIS PRECISO (Trocado de 'yolov8n' para 'yolov8m')
# Se o seu PC não tiver GPU e travar, use o 'yolov8s-seg.pt' (Small), que é o meio-termo.
model = YOLO('yolov8n-seg.pt')

nomes_classes = model.names
cores_por_id = {}

# 2. PESO DAS CORES AJUSTADO (Transparência da máscara)
# 0.4 significa 40% da cor da máscara e 60% da imagem real da webcam.
# Isso deixa a máscara suave e permite ver o celular/pessoa por trás.
peso_cor = 0.4

cap = cv2.VideoCapture(0)

# Opcional: Forçar a webcam a capturar em maior resolução (Melhora a precisão)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 3. PARÂMETROS DE ALTA PRECISÃO
    # conf=0.3: Filtra detecções fracas (evita que o modelo confunda sua mão com um telefone)
    # iou=0.5: Melhora o rastreamento quando a mão fica na frente do telefone
    # imgsz=640: Garante que o YOLO processe a imagem em boa resolução
    results = model.track(
        frame, 
        persist=True, 
        stream=True, 
        verbose=False, 
        classes=[0, 67],
        conf=0.3,
        iou=0.5,
        imgsz=640
    )
    
    for r in results:
        if r.masks is not None and r.boxes.id is not None:
            
            mascaras = r.masks.xy
            boxes = r.boxes
            
            for segmento, box in zip(mascaras, boxes):
                class_id = int(box.cls.item())
                track_id = int(box.id.item())
                confianca = float(box.conf.item())
                
                nome_classe = nomes_classes[class_id]
                
                if track_id not in cores_por_id:
                    cores_por_id[track_id] = (
                        random.randint(50, 255), 
                        random.randint(50, 255), 
                        random.randint(50, 255)
                    )
                
                cor = cores_por_id[track_id]
                
                # --- DESENHAR A MÁSCARA SEMI-TRANSPARENTE ---
                pts = np.array(segmento, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], cor)
                
                # Aplica o peso_cor atualizado para dar o efeito translúcido
                cv2.addWeighted(overlay, peso_cor, frame, 1 - peso_cor, 0, frame)
                
                # --- DESENHAR A CAIXA E O RÓTULO DINÂMICO ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                
                texto = f"{nome_classe} {track_id} [{confianca:.2f}]"
                
                (largura_texto, altura_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - altura_texto - 10), (x1 + largura_texto, y1), cor, -1)
                
                # Texto em branco por cima do fundo colorido
                cv2.putText(frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Segmentacao de Alta Precisao", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()