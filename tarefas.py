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