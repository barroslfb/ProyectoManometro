import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
import cv2
import math
import numpy as np
import re
import easyocr
from ultralytics import YOLO

print("[INFO] Carregando motor OCR (EasyOCR)...")
reader = easyocr.Reader(['en'], gpu=True) # Mude gpu=False se não tiver placa de vídeo compatível

# ==========================================
# VARIÁVEIS DE ESTADO (CACHE DE CALIBRAÇÃO)
# ==========================================
CALIBRADO = False
VALOR_MINIMO = 0.0
VALOR_MAXIMO = 100.0
TENTATIVAS_OCR = 0
MAX_TENTATIVAS = 50

def extrair_numero_imagem(frame, punto, centro, nome_janela="Crop OCR", pad=45, deslocamento=25):
    """
    Recorta a imagem com deslocamento vetorial para o centro e aplica OCR.
    """
    px, py = int(punto[0]), int(punto[1])
    cx, cy = int(centro[0]), int(centro[1])
    h, w = frame.shape[:2]

    # Deslocamento Radial
    dx = cx - px
    dy = cy - py
    distancia = math.hypot(dx, dy)

    if distancia > 0:
        fator_x = (dx / distancia) * deslocamento
        fator_y = (dy / distancia) * deslocamento
    else:
        fator_x, fator_y = 0, 0

    alvo_x = int(px + fator_x)
    alvo_y = int(py + fator_y)

    x1, y1 = max(0, alvo_x - pad), max(0, alvo_y - pad)
    x2, y2 = min(w, alvo_x + pad), min(h, alvo_y + pad)
    
    recorte = frame[y1:y2, x1:x2]
    if recorte.size == 0: return None

# Pré-processamento
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    cinza_ampliado = cv2.resize(cinza, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow(nome_janela, cinza_ampliado)
    
    # ==========================================
    # 2. OCR COM FILTRO DE PROXIMIDADE ESPACIAL
    # ==========================================
    resultados_ocr = reader.readtext(cinza_ampliado, allowlist='0123456789.-')
    
    melhor_numero = None
    menor_dist_centro = float('inf')
    centro_recorte = (pad * 2, pad * 2) 

    for (bbox, texto, confianca) in resultados_ocr:
        if confianca > 0.4:
            numeros = re.findall(r"[-+]?\d*\.\d+|\d+", texto)
            if numeros:
                box_x_centro = (bbox[0][0] + bbox[2][0]) / 2
                box_y_centro = (bbox[0][1] + bbox[2][1]) / 2
                dist = math.hypot(box_x_centro - centro_recorte[0], box_y_centro - centro_recorte[1])
                
                if dist < menor_dist_centro:
                    menor_dist_centro = dist
                    melhor_numero = float(numeros[0])
                    
    return melhor_numero

def calcular_angulo(centro, punto):
    x_c, y_c = centro
    x_p, y_p = punto
    radianes = math.atan2(y_c - y_p, x_p - x_c)
    grados = math.degrees(radianes)
    if grados < 0:
        grados += 360
    return grados

def extrair_lectura(puntos, val_min, val_max):
    p_minimo, p_maximo, aguja, centro = puntos[0], puntos[1], puntos[2], puntos[3]

    ang_min = calcular_angulo(centro, p_minimo)
    ang_max = calcular_angulo(centro, p_maximo)
    ang_aguja = calcular_angulo(centro, aguja)
    
    arco_total = ang_min - ang_max
    if arco_total < 0: arco_total += 360
        
    posicion_aguja = ang_min - ang_aguja
    if posicion_aguja < 0: posicion_aguja += 360

    if posicion_aguja > arco_total:
        if posicion_aguja >= 300: 
            posicion_aguja = 0.0
        else:
            posicion_aguja = arco_total

    porcentaje = posicion_aguja / arco_total
    porcentaje = max(0.0, min(1.0, porcentaje)) 
    
    return val_min + (porcentaje * (val_max - val_min))

def iniciar_webcam(ruta_modelo):
    global CALIBRADO, VALOR_MINIMO, VALOR_MAXIMO, TENTATIVAS_OCR
    
    print("[INFO] Cargando YOLO...")
    model = YOLO(ruta_modelo)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No fue posible acceder a la webcam.")
        return

    while True:
        exito, frame = cap.read()
        if not exito: break
            
        resultados = model.predict(source=frame, verbose=False)
        resultado = resultados[0]

        if resultado.keypoints is not None and len(resultado.keypoints.xy) > 0:
            puntos_detectados = resultado.keypoints.xy[0].cpu().numpy()
            
            if len(puntos_detectados) >= 4 and not np.any(np.all(puntos_detectados == 0, axis=1)):
                
                # ====================================================
                # ESTADO 1: CALIBRAÇÃO HÍBRIDA (SÓ LÊ O MAX)
                # ====================================================
                if not CALIBRADO and TENTATIVAS_OCR < MAX_TENTATIVAS:
                    cv2.putText(frame, "CALIBRANDO ESCALA MAX (OCR)...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    
                    VALOR_MINIMO = 0.0 
                    max_lido = extrair_numero_imagem(frame, puntos_detectados[1], puntos_detectados[3], "Visao OCR - MAX", pad=45, deslocamento=15)
                    
                    if max_lido is not None and max_lido > VALOR_MINIMO:
                        if max_lido <= 1000.0: # Ignora erros de leitura bizarros
                            VALOR_MAXIMO = max_lido
                            CALIBRADO = True
                            try:
                                cv2.destroyWindow("Visao OCR - MAX") 
                            except:
                                pass
                            print(f"[SUCESSO] Calibrado: Min={VALOR_MINIMO}, Max={VALOR_MAXIMO}")
                    else:
                        TENTATIVAS_OCR += 1
                        
                elif not CALIBRADO and TENTATIVAS_OCR >= MAX_TENTATIVAS:
                    cv2.putText(frame, "FALHA OCR - USANDO PADRAO (0-100)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    VALOR_MINIMO = 0.0
                    VALOR_MAXIMO = 100.0
                    CALIBRADO = True
                    try:
                        cv2.destroyWindow("Visao OCR - MAX")
                    except:
                        pass

                # ====================================================
                # ESTADO 2: MONITORAMENTO
                # ====================================================
                elif CALIBRADO:
                    try:
                        lectura_final = extrair_lectura(puntos_detectados, VALOR_MINIMO, VALOR_MAXIMO)
                        texto_lectura = f"{lectura_final:.1f} (Escala: 0 - {VALOR_MAXIMO})"
                        
                        cv2.rectangle(frame, (10, 10), (600, 70), (0, 0, 0), -1)
                        cv2.putText(frame, f"VALOR: {texto_lectura}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        colores = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]
                        for i, punto in enumerate(puntos_detectados):
                            x, y = int(punto[0]), int(punto[1])
                            cv2.circle(frame, (x, y), 6, colores[i], -1)
                            if i == 2: 
                                cx, cy = int(puntos_detectados[3][0]), int(puntos_detectados[3][1])
                                cv2.line(frame, (cx, cy), (x, y), (255,0,0), 2)
                    except Exception:
                        pass
            else:
                 cv2.putText(frame, "Buscando manometro...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Lector Industrial - Calibracao Dinamica', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_webcam("bestAnalogKeyPoint.pt")