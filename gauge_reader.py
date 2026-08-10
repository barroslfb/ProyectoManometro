import argparse
import os
from dotenv import load_dotenv

load_dotenv()

import re
import cv2
import numpy as np
from collections import defaultdict
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pytesseract
import urllib.request
import json
import threading
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS

import base64
import time
import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import PublishToIoTCoreRequest, QOS
import boto3

# ==============================================================================
# AWS S3 CONFIGURATION
# ==============================================================================
s3_client = None

def get_s3_client():
    try:
        return boto3.client('s3', region_name='us-east-1',
                            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    except Exception as e:
        print(f"Error fallback boto3: {e}")
        return None

def upload_image_to_s3(frame):
    global s3_client
    if not s3_client:
        s3_client = get_s3_client()
    if not s3_client: return
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        s3_client.put_object(
            Bucket='proyeco-manometro',
            Key='imagenes/latest.jpg',
            Body=buffer.tobytes(),
            ContentType='image/jpeg'
        )
    except Exception as e:
        print(f"[ERROR S3] Fallo al subir imagen: {e}")
        s3_client = None

# ==============================================================================
# FLASK WEB SERVER CONFIGURATION
# ==============================================================================
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
CORS(app)

current_frame = None
current_pressure = None
frame_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    global current_pressure
    return jsonify({'pressure': current_pressure})

def start_web_server():
    print("[INFO] Iniciando servidor Web en el puerto 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def enviar_alerta_resend(presion, frame):
    print(f"\n[ALERTA] Presion alta detectada ({presion:.2f} PSI). Enviando correo via Resend...")
    
    _, buffer = cv2.imencode('.jpg', frame)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}",
        "Content-Type": "application/json",
        "User-Agent": "Resend/1.0.0 (Python)"
    }
    data = {
        "from": "onboarding@resend.dev",
        "to": ["luizfelipepab@hotmail.com"],
        "subject": "ALERTA: Presion excesiva en Manometro",
        "html": f"<strong>Peligro:</strong> La presion actual es de <strong>{presion:.2f} PSI</strong>, superando el limite de 10 PSI. Favor de verificar el equipo adjunto.",
        "attachments": [
            {
                "filename": "evidencia_presion.jpg",
                "content": img_b64
            }
        ]
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            print("[ALERTA] Correo enviado exitosamente.")
    except Exception as e:
        print(f"[ERROR ALERTA] Fallo al enviar el correo: {e}")

# ==============================================================================
# AWS IOT GREENGRASS IPC CONFIGURATION
# ==============================================================================
try:
    ipc_client = awsiot.greengrasscoreipc.connect()
    print("[INFO] Conectado a Greengrass IPC!")
except Exception as e:
    print(f"[WARNING] Error al conectar a Greengrass IPC: {e}. Ejecutando offline.")
    ipc_client = None

def enviar_a_aws(presion_leida):
    if not ipc_client:
        return
    
    topico = "manometro/dados" 
    payload = {
        "timestamp": int(time.time()),
        "presion_psi": round(presion_leida, 2),
        "dispositivo": "Jetson_1"
    }

    try:
        request = PublishToIoTCoreRequest()
        request.topic_name = topico
        request.payload = bytes(json.dumps(payload), "utf-8")
        request.qos = QOS.AT_LEAST_ONCE
        
        operation = ipc_client.new_publish_to_iot_core()
        operation.activate(request)
        future = operation.get_response()
        future.result(timeout=5.0)
    except Exception as e:
        print(f"[ERROR AWS] Fallo al publicar: {e}")

is_inferring = True

import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import IoTCoreMessage, SubscribeToIoTCoreRequest, QOS

class StreamHandler(client.SubscribeToIoTCoreStreamHandler):
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: IoTCoreMessage) -> None:
        global is_inferring
        try:
            message = str(event.message.payload, "utf-8")
            data = json.loads(message)
            if "ligar" in data:
                is_inferring = bool(data["ligar"])
                print(f"\n[MQTT] Comando recibido: LIGAR = {is_inferring}", flush=True)
            elif "encender" in data:
                is_inferring = bool(data["encender"])
                print(f"\n[MQTT] Comando recibido: ENCENDER = {is_inferring}", flush=True)
        except Exception as e:
            print(f"[MQTT] Error al analizar el mensaje: {e}", flush=True)

    def on_stream_error(self, error: Exception) -> bool:
        return False

    def on_stream_closed(self) -> None:
        pass

mqtt_operation = None

def setup_mqtt_subscription():
    global mqtt_operation
    if not ipc_client:
        return
    try:
        request = SubscribeToIoTCoreRequest()
        request.topic_name = "manometro/controle"
        request.qos = QOS.AT_MOST_ONCE
        handler = StreamHandler()
        mqtt_operation = ipc_client.new_subscribe_to_iot_core(handler)
        future = mqtt_operation.activate(request)
        future.result(timeout=5.0)
        print("[INFO] Inscrito en el topico manometro/controle con exito!", flush=True)
    except Exception as e:
        print(f"[ERROR AWS] Fallo al inscribir en el topico: {e}", flush=True)

# ==============================================================================
# CONFIGURACIONES GLOBALES E INICIALIZACION TARDIA
# ==============================================================================
CONF_THRESHOLD = 0.15  
CALIBRATION_FRAMES = 100
SAVE_DIR = "frames_manometro_registrados"

CLASSES = {
    0: "below-max-value", 1: "below-min-value", 2: "below-unit",
    3: "center", 4: "gauge", 5: "max-value", 6: "min-value",
    7: "needle", 8: "tip", 9: "unit", 10: "upper-max-value",
    11: "upper-min-value", 12: "upper-unit"
}



# ==============================================================================
# CLASSE TENSORRT
# ==============================================================================
class TRTYOLOv8Seg:
    def __init__(self, engine_path, input_shape=(640, 640)):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.input_shape = input_shape
        self.engine_path = engine_path
        
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})
                
        return inputs, outputs, bindings, stream

    def infer(self, img_bgr):
        orig_h, orig_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, self.input_shape)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_tensor = np.ascontiguousarray(img_tensor)

        np.copyto(self.inputs[0]["host"], img_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        if self.outputs[0]["host"].size == 819200:
            protos_host = self.outputs[0]["host"]
            preds_host = self.outputs[1]["host"]
        else:
            preds_host = self.outputs[0]["host"]
            protos_host = self.outputs[1]["host"]

        preds = preds_host.reshape(1, -1, 8400)[0].T
        protos = protos_host.reshape(32, 160, 160)

        boxes_by_class = defaultdict(list)
        masks_by_class = defaultdict(list)
        
        boxes_xywh = preds[:, :4]
        mask_coeffs = preds[:, -32:]
        scores = preds[:, 4:-32]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask_conf = confidences > CONF_THRESHOLD
        boxes_xywh = boxes_xywh[mask_conf]
        confidences = confidences[mask_conf]
        class_ids = class_ids[mask_conf]
        mask_coeffs = mask_coeffs[mask_conf]

        if len(boxes_xywh) == 0:
            return boxes_by_class, masks_by_class

        x_c, y_c, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes_xyxy = np.column_stack([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2])

        rx, ry = orig_w / self.input_shape[0], orig_h / self.input_shape[1]
        boxes_xyxy[:, [0, 2]] *= rx
        boxes_xyxy[:, [1, 3]] *= ry

        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), CONF_THRESHOLD, 0.45)
        
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                cls_id = class_ids[idx]
                cname = CLASSES.get(cls_id, f"cls_{cls_id}")
                box = boxes_xyxy[idx]
                conf = confidences[idx]
                boxes_by_class[cname].append((box, conf))

                coeff = mask_coeffs[idx]
                mask = np.tensordot(coeff, protos, axes=([0], [0]))
                mask = 1.0 / (1.0 + np.exp(-mask))
                
                mask = cv2.resize(mask, (orig_w, orig_h))
                mask = (mask > 0.5).astype(np.uint8)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    masks_by_class[cname].append(np.vstack(contours).squeeze())

        return boxes_by_class, masks_by_class

# ==============================================================================
# FUNCIONES GEOMETRICAS Y PARSEO NUMERICO
# ==============================================================================
def point_from_box(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])

def get_needle_tip(center_point, boxes_by_class, masks_by_class):
    if "tip" in boxes_by_class and len(boxes_by_class["tip"]) > 0:
        idx = np.argmax([x[1] for x in boxes_by_class["tip"]])
        return point_from_box(boxes_by_class["tip"][idx][0])

    if "needle" in masks_by_class and len(masks_by_class["needle"]) > 0:
        idx = np.argmax([x[1] for x in boxes_by_class["needle"]])
        mask_points = masks_by_class["needle"][idx] 
        if mask_points is not None and len(mask_points) > 0 and center_point is not None:
            if len(mask_points.shape) == 1:
                mask_points = mask_points.reshape(-1, 2)
            distances = np.linalg.norm(mask_points - center_point, axis=1)
            max_idx = np.argmax(distances)
            return mask_points[max_idx]
    return None

def calculate_value(center, needle_tip, min_pos, max_pos, max_val, min_val):
    if any(p is None for p in (center, needle_tip, min_pos, max_pos)) or max_val is None or min_val is None:
        return None
        
    ang_min = np.degrees(np.arctan2(-(min_pos[1] - center[1]), min_pos[0] - center[0])) % 360
    ang_max = np.degrees(np.arctan2(-(max_pos[1] - center[1]), max_pos[0] - center[0])) % 360
    ang_needle = np.degrees(np.arctan2(-(needle_tip[1] - center[1]), needle_tip[0] - center[0])) % 360

    sweep_total = (ang_min - ang_max) % 360
    sweep_needle = (ang_min - ang_needle) % 360

    if sweep_total == 0: return min_val
    fraction_clamped = max(0.0, min(1.0, sweep_needle / sweep_total))
    return min_val + fraction_clamped * (max_val - min_val)

def parse_ocr_number(text):
    if not text: return None
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.strip().replace(",", "."))
    return float(numbers[0]) if numbers else None

# ==============================================================================
# PIPELINE EJECUCION PRINCIPAL
# ==============================================================================
def main(args):
    global current_frame, current_pressure
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = TRTYOLOv8Seg(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"Fallo al abrir la fuente: {source}")

    best_max_crop = {"conf": -1.0, "img": None}
    best_min_crop = {"conf": -1.0, "img": None}
    
    physical_max_val = None
    physical_min_val = None
    
    last_min_pt = None
    last_max_pt = None

    frame_count = 0
    print("[INFO] Iniciando captura de escala unica...")
    
    has_display = "DISPLAY" in os.environ
    if not has_display:
        print("[INFO] Ejecucion sin interfaz grafica (Headless mode). Las lecturas se imprimiran en consola.", flush=True)

    alert_sent = False
    last_aws_send_time = 0.0

    # Start Flask Web Server in a background thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    try:
        setup_mqtt_subscription()
        while True:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str):
                    print("[INFO] Video finalizado. Reiniciando el loop...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            val = None

            if not is_inferring:
                txt = "PAUSADO - ESPERANDO COMANDO MQTT"
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                
                with frame_lock:
                    current_frame = annotated_frame.copy()
                
                current_t = time.time()
                if current_t - last_aws_send_time >= 1.0:
                    upload_image_to_s3(annotated_frame)
                    last_aws_send_time = current_t
                
                time.sleep(0.1)
                continue
            
            frame_count += 1
            h, w = frame.shape[:2]
            annotated_frame = frame.copy()
            
            boxes_by_class, masks_by_class = model.infer(frame)
    
            center_box = max(boxes_by_class["center"], key=lambda x: x[1])[0] if "center" in boxes_by_class else None
            center_pt = point_from_box(center_box) if center_box is not None else None
            needle_pt = get_needle_tip(center_pt, boxes_by_class, masks_by_class)
    
            for cname, det_list in boxes_by_class.items():
                for (box, conf) in det_list:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(annotated_frame, f"{cname} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    
            # ETAPA A: CALIBRACION (Recorte de max_value y min_value)
            if frame_count <= CALIBRATION_FRAMES:
                padding = 5
                
                if "max-value" in boxes_by_class:
                    box, conf = max(boxes_by_class["max-value"], key=lambda x: x[1])
                    if conf > best_max_crop["conf"]:
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[max(0, y1-padding):min(h, y2+padding), max(0, x1-padding):min(w, x2+padding)]
                        if crop.size > 0:
                            # Ampliacion simple para mejor visualizacion por las capas convolucionales
                            scaled = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            best_max_crop = {"conf": conf, "img": scaled.copy()}
                        
                if "min-value" in boxes_by_class:
                    box, conf = max(boxes_by_class["min-value"], key=lambda x: x[1])
                    if conf > best_min_crop["conf"]:
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[max(0, y1-padding):min(h, y2+padding), max(0, x1-padding):min(w, x2+padding)]
                        if crop.size > 0:
                            scaled = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            best_min_crop = {"conf": conf, "img": scaled.copy()}
    
                cv2.putText(annotated_frame, f"Calibrando: {frame_count}/{CALIBRATION_FRAMES}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                
                if frame_count == CALIBRATION_FRAMES:
                    print("\n[INFO] Activando Tesseract OCR. Bajo consumo de memoria...")
                    
                    # Configuracion de tesseract para enfocar en numeros
                    tess_config = '--psm 6 -c tessedit_char_whitelist=0123456789.-'
                    
                    # Lectura MAX VALUE
                    if best_max_crop["img"] is not None:
                        cv2.imwrite(os.path.join(SAVE_DIR, "tesseract_max_value.png"), best_max_crop["img"])
                        ocr_text = pytesseract.image_to_string(best_max_crop["img"], config=tess_config).strip()
                        if ocr_text:
                            physical_max_val = parse_ocr_number(ocr_text)
                            print(f" > Max Value leido: '{ocr_text}' -> Formateado: {physical_max_val}")
                        
                        if physical_max_val is None:
                            print(" > [FALLBACK] Fallo en OCR. Forzando MAX = 60.0")
                            physical_max_val = 60.0
                    
                    # Lectura MIN VALUE
                    if best_min_crop["img"] is not None:
                        cv2.imwrite(os.path.join(SAVE_DIR, "tesseract_min_value.png"), best_min_crop["img"])
                        ocr_text = pytesseract.image_to_string(best_min_crop["img"], config=tess_config).strip()
                        if ocr_text:
                            physical_min_val = parse_ocr_number(ocr_text)
                            print(f" > Min Value leido: '{ocr_text}' -> Formateado: {physical_min_val}")
                        
                        if physical_min_val is None:
                            print(" > [FALLBACK] Fallo en OCR. Forzando MIN = 0.0")
                            physical_min_val = 0.0
    
            # ETAPA B: LECTURA GEOMETRICA
            else:
                if "min-value" in boxes_by_class:
                    last_min_pt = point_from_box(max(boxes_by_class["min-value"], key=lambda x: x[1])[0])
                if "max-value" in boxes_by_class:
                    last_max_pt = point_from_box(max(boxes_by_class["max-value"], key=lambda x: x[1])[0])
    
                val = calculate_value(center_pt, needle_pt, last_min_pt, last_max_pt, physical_max_val, physical_min_val)
                
                txt = f"PRESION: {val:.2f}" if val is not None else "Esperando geometria de la aguja..."
                # Dibuja el texto con un borde (outline) negro para que sea legible sin necesitar la barra negra que corta el video
                cv2.putText(annotated_frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            if val is not None:
                if not has_display and frame_count % 30 == 0:
                    print(f"[HEADLESS] Frame {frame_count} - Lectura de Presion: {val:.2f}", flush=True)

                current_t = time.time()
                if current_t - last_aws_send_time >= 1.0:
                    enviar_a_aws(val)
                    upload_image_to_s3(annotated_frame)
                    last_aws_send_time = current_t

                # Logica Anti-Spam para Resend (>10 PSI)
                if val > 10.0 and not alert_sent:
                    enviar_alerta_resend(val, annotated_frame)
                    alert_sent = True
                elif val <= 10.0 and alert_sent:
                    alert_sent = False
                    print(f"\n[INFO] Presion normalizada ({val:.2f} PSI). Alerta reseteada.")
    
            if has_display:
                cv2.imshow("TRT + Tesseract", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            # Update global state for Web Server
            with frame_lock:
                current_frame = annotated_frame.copy()
                current_pressure = val
    
    except Exception as e:
        print(f"\n[ERROR] Excepcion capturada: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Ejecucion interrumpida por el usuario (Ctrl+C).")
    finally:
        print("[INFO] Limpiando recursos y cerrando PyCUDA...")
        cap.release()
        if has_display:
            cv2.destroyAllWindows()
        import pycuda.autoinit
        if pycuda.autoinit.context:
            try:
                pycuda.autoinit.context.pop()
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()
    main(args)
