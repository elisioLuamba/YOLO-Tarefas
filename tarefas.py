import cv2
import numpy as np
import random
from ultralytics import YOLO

# 1. MODELO (Mude para 'yolov8m-seg.pt' se tiver GPU e quiser mais precisão no cenário)
model = YOLO('yolov8n-seg.pt')

nomes_classes = model.names

# Dicionário para salvar uma cor fixa para cada CLASSE (Segmentação Semântica)
# Ex: Toda cadeira terá a mesma cor, toda mesa terá a mesma cor, etc.
cores_por_classe = {}

# Transparência da máscara (0.35 para destacar bem os objetos do cenário)
peso_cor = 0.35

cap = cv2.VideoCapture(0)

# Resolução da webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Máscara preta acumuladora para desenhar todas as segmentações do frame
    mascara_acumulada = np.zeros_like(frame)
    
    # 2. DETECTAR TUDO (Removido o filtro 'classes=[0, 67]')
    results = model.predict(
        frame, 
        stream=True, 
        verbose=False, 
        conf=0.25,  # Reduzido levemente para detectar objetos menores ao fundo (cadeiras/mesas)
        imgsz=640
    )
    
    for r in results:
        if r.masks is not None:
            mascaras = r.masks.xy
            boxes = r.boxes
            
            for segmento, box in zip(mascaras, boxes):
                class_id = int(box.cls.item())
                confianca = float(box.conf.item())
                nome_classe = nomes_classes[class_id]
                
                # Se a classe ainda não tem cor definida, gera uma cor fixa para ela
                if class_id not in cores_por_classe:
                    cores_por_classe[class_id] = (
                        random.randint(40, 255), 
                        random.randint(40, 255), 
                        random.randint(40, 255)
                    )
                
                cor = cores_por_classe[class_id]
                
                # --- DESENHAR A MÁSCARA NO ACUMULADOR ---
                pts = np.array(segmento, dtype=np.int32)
                cv2.fillPoly(mascara_acumulada, [pts], cor)
                
                # --- RÓTULO DA CLASSE ---
                x1, y1, _, _ = map(int, box.xyxy[0])
                texto = f"{nome_classe} [{confianca:.2f}]"
                
                (largura_texto, altura_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Garante que o texto não saia da tela na parte superior
                y1_ajustado = max(y1, altura_texto + 10)
                
                cv2.rectangle(frame, (x1, y1_ajustado - altura_texto - 10), (x1 + largura_texto, y1_ajustado), cor, -1)
                cv2.putText(frame, texto, (x1, y1_ajustado - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # --- APLICAR MÁSCARAS SEMÂNTICAS UNIFICADAS ---
    indices_mascara = mascara_acumulada > 0
    overlay = frame.copy()
    overlay[indices_mascara] = mascara_acumulada[indices_mascara]
    
    cv2.addWeighted(overlay, peso_cor, frame, 1 - peso_cor, 0, frame)

    cv2.imshow("Segmentacao Semantica Total", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()