from ultralytics import YOLO
import cv2

# ==============================================================================
# ETAPA 1: IMPORTAÇÃO E INICIALIZAÇÃO DO MODELO DE IA
# ==============================================================================
# Carregamos o modelo pré-treinado YOLO nano (yolov8n.pt).
# Por ser leve, ele garante alta taxa de quadros por segundo (FPS) em tempo real.
# Se for a primeira execução, o download dos pesos (.pt) será feito automaticamente.
model = YOLO("yolov8n.pt")


# ==============================================================================
# ETAPA 2: CAPTURA E VALIDAÇÃO DO HARDWARE (WEBCAM)
# ==============================================================================
# Inicializa o objeto de captura de vídeo. O argumento 0 aponta para a webcam padrão.
cap = cv2.VideoCapture(0)

# Validação crucial: se a câmera estiver ocupada ou desconectada, o script para aqui
if not cap.isOpened():
    print("Erro crítico: Não foi possível acessar a webcam.")
    print("Verifique se ela está conectada ou sendo usada por outro app.")
    exit()


# ==============================================================================
# ETAPA 3: LOOP PRINCIPAL DE PROCESSAMENTO E INFERÊNCIA
# ==============================================================================
print("Sistema iniciado com sucesso. Pressione 'q' na janela do vídeo para sair.")

while True:
    # 3.1 Captura o frame atual enviado pela webcam
    ret, frame = cap.read()
    if not ret:
        print("Aviso: Falha ao receber frame da câmera. Encerrando loop...")
        break

    # 3.2 Processamento da IA (Inferência)
    # stream=True otimiza o uso de memória RAM ao processar o fluxo de vídeo
    results = model(frame, stream=True)

    # 3.3 Renderização dos Resultados
    # Percorremos os resultados e usamos o .plot() para desenhar caixas e rótulos
    for r in results:
        frame = r.plot()

    # 3.4 Exibição Visual
    # Abre a janela nativa do OpenCV e exibe o frame já modificado pela IA
    cv2.imshow("YOLO - Curso de Visao Computacional", frame)

    # 3.5 Controle de Interrupção
    # Aguarda 1 milissegundo por comando do teclado. Se for a tecla 'q', quebra o loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==============================================================================
# ETAPA 4: DESALOCAÇÃO DE RECURSOS E LIMPEZA DE MEMÓRIA
# ==============================================================================
# Libera o hardware da webcam para que outros programas do sistema possam usá-la
cap.release()

# Sequência de segurança para fechar todas as janelas do OpenCV sem travar o Python
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Script encerrado de forma limpa e segura.")