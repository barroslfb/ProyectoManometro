import cv2

import math

import numpy as np

from ultralytics import YOLO



# ==========================================

# CONFIGURACIONES DE TU MANÓMETRO FÍSICO

# ==========================================

VALOR_MINIMO = 0.0

VALOR_MAXIMO = 100.0



def calcular_angulo(centro, punto):

    """Calcula el ángulo en grados entre el centro y un punto específico."""

    x_c, y_c = centro

    x_p, y_p = punto

    radianes = math.atan2(y_c - y_p, x_p - x_c)

    grados = math.degrees(radianes)

    if grados < 0:

        grados += 360

    return grados



def extraer_lectura(puntos):

    """Aplica la lógica geométrica para encontrar el valor numérico con filtro de jitter."""

    p_minimo = puntos[0]  # Verde

    p_maximo = puntos[1]  # Rojo

    aguja = puntos[2]     # Azul

    centro = puntos[3]    # Amarillo



    ang_min = calcular_angulo(centro, p_minimo)

    ang_max = calcular_angulo(centro, p_maximo)

    ang_aguja = calcular_angulo(centro, aguja)

   

    arco_total = ang_min - ang_max

    if arco_total < 0: arco_total += 360

       

    posicion_aguja = ang_min - ang_aguja

    if posicion_aguja < 0: posicion_aguja += 360



    # ==========================================

    # CORRECCIÓN DE "WRAP-AROUND" (JITTER DE IA)

    # ==========================================

    if posicion_aguja > arco_total:

        # Si el valor es muy cercano a 360 (ej: 358), significa que la aguja

        # está oscilando milimétricamente detrás del punto cero.

        if posicion_aguja >= 300:

            posicion_aguja = 0.0

        # De lo contrario, si pasó el arco total pero está lejos de 360,

        # significa que pasó levemente del valor máximo.

        else:

            posicion_aguja = arco_total

    # ==========================================



    porcentaje = posicion_aguja / arco_total

   

    # La traba de seguridad se mantiene por precaución

    porcentaje = max(0.0, min(1.0, porcentaje))

   

    lectura = VALOR_MINIMO + (porcentaje * (VALOR_MAXIMO - VALOR_MINIMO))

    return lectura



# ==========================================

# MOTOR DE INFERENCIA EN TIEMPO REAL

# ==========================================

def iniciar_webcam(ruta_modelo):

    print("[INFO] Cargando la Inteligencia Artificial...")

    model = YOLO(ruta_modelo)

   

    print("[INFO] Iniciando la Webcam. Presione 'Q' para salir.")

    # Inicia la captura de video. El '0' representa la cámara por defecto del notebook/PC.

    # Si tienes más de una cámara conectada, intenta cambiar a 1, 2, etc.

    cap = cv2.VideoCapture(0)



    # Verifica si la cámara se abrió con éxito

    if not cap.isOpened():

        print("[ERROR] No fue posible acceder a la webcam.")

        return



    # Bucle infinito para procesar el video fotograma a fotograma

    while True:

        exito, frame = cap.read()

        if not exito:

            print("[ERROR] Falla al leer el fotograma de la cámara.")

            break

           

        # Refleja el fotograma horizontalmente (opcional, solo para no confundir en pantalla)

        # frame = cv2.flip(frame, 1)



        # Ejecuta YOLO en el fotograma actual.

        # verbose=False desactiva el spam de texto en el terminal para cada fotograma procesado.

        resultados = model.predict(source=frame, verbose=False)

        resultado = resultados[0]



        # Verifica si el modelo encontró algo y si tiene la capa de keypoints

        if resultado.keypoints is not None and len(resultado.keypoints.xy) > 0:

            puntos_detectados = resultado.keypoints.xy[0].cpu().numpy()

           

            # Filtro de Seguridad Robusto:

            # Verifica si existen 4 puntos Y si ninguno de ellos es [0, 0] (YOLO devuelve cero cuando la mano tapa un punto)

            if len(puntos_detectados) >= 4 and not np.any(np.all(puntos_detectados == 0, axis=1)):

                try:

                    lectura_final = extraer_lectura(puntos_detectados)

                   

                    # Formato visual en pantalla (HUD)

                    texto_lectura = f"{lectura_final:.1f} PSI" # Cambia PSI por tu unidad

                   

                    # Dibuja un fondo negro para que el texto sea legible

                    cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)

                    cv2.putText(frame, f"LECTURA: {texto_lectura}", (20, 50),

                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)



                    # Dibuja los puntos físicos en el manómetro

                    colores = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]

                    for i, punto in enumerate(puntos_detectados):

                        x, y = int(punto[0]), int(punto[1])

                        cv2.circle(frame, (x, y), 6, colores[i], -1)

                       

                        # Une el centro (índice 3) a la punta de la aguja (índice 2) con una línea AZUL

                        if i == 2:

                            cx, cy = int(puntos_detectados[3][0]), int(puntos_detectados[3][1])

                            cv2.line(frame, (cx, cy), (x, y), (255,0,0), 2)

                           

                except Exception as e:

                    # Si hay un fallo matemático grotesco, ignora el fotograma para no trabar el video

                    pass

            else:

                 cv2.putText(frame, "Buscando manometro completo...", (20, 50),

                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        # Muestra el resultado en la ventana

        cv2.imshow('Lector Industrial de Manometros - YOLOv8', frame)



        # Espera 1 milisegundo y verifica si el usuario presionó la tecla 'q' para cerrar

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break



    # Limpieza y apagado de la cámara

    cap.release()

    cv2.destroyAllWindows()



# ==========================================

# INICIO DEL PROGRAMA

# ==========================================

if __name__ == "__main__":

    # Asegúrate de que el archivo best.pt esté en la misma carpeta del script

    iniciar_webcam("bestAnalogKeyPoint.pt") 

