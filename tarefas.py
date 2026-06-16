import cv2
import numpy as np
import random
from ultralytics import YOLO

# 1. Carregar o modelo pré-treinado de SEGMENTAÇÃO
model = YOLO('yolov8n-seg.pt')

# Dicionário para armazenar as cores geradas para cada ID
cores_por_id = {}

# Peso da cor da máscara (0.0 a 1.0) - Aumentado para cores mais vivas
peso_cor = 0.7

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 2. Rastreamento para manter a consistência das detecções
    results = model.track(frame, persist=True, classes=0, stream=True, verbose=False)
    
    for r in results:
        if r.masks is not None and r.boxes.id is not None:
            
            mascaras = r.masks.xy
            boxes = r.boxes
            
            for segmento, box in zip(mascaras, boxes):
                track_id = int(box.id.item())
                confianca = float(box.conf.item())
                
                # Gera e armazena uma cor forte se o ID for novo
                if track_id not in cores_por_id:
                    cores_por_id[track_id] = (
                        random.randint(50, 255), 
                        random.randint(50, 255), 
                        random.randint(50, 255)
                    )
                
                cor = cores_por_id[track_id]
                
                # --- DESENHAR A MÁSCARA COM MAIS PESO ---
                pts = np.array(segmento, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], cor)
                
                # Aplica a transparência: 70% da cor da máscara, 30% da imagem original
                cv2.addWeighted(overlay, peso_cor, frame, 1 - peso_cor, 0, frame)
                
                # --- DESENHAR A CAIXA E O NOME (RÓTULO) ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                
                # Texto com Nome, ID e Confiança
                texto = f"Pessoa {track_id} {confianca:.2f}"
                
                # Criar um fundo preenchido para o texto (melhora a leitura, igual ao YOLO)
                (largura_texto, altura_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - altura_texto - 10), (x1 + largura_texto, y1), cor, -1)
                
                # Inserir o texto em branco ou preto dependendo da cor para dar contraste
                cv2.putText(frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Segmentacao de Pessoas", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()