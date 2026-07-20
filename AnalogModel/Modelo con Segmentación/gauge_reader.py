import argparse
import os
import re
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Inicialização tardia do EasyOCR para otimizar o tempo de inicialização do script
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # Configurado para inglês/números. Mude para gpu=True se tiver CUDA disponível
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
    return _ocr_reader


# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================
CONF_THRESHOLD = 0.35      # Confiança mínima para o YOLOv8-seg
CALIBRATION_FRAMES = 100   # Quantidade exata de frames para selecionar o melhor crop
SAVE_DIR = "frames_manometro_registrados"

SCALES_CONFIG = {
    "principal": {"prefix": "", "max_val": None, "unit": "unit", "last_min_pt": None, "last_max_pt": None},
    "upper": {"prefix": "upper-", "max_val": None, "unit": "upper-unit", "last_min_pt": None, "last_max_pt": None},
    "below": {"prefix": "below-", "max_val": None, "unit": "below-unit", "last_min_pt": None, "last_max_pt": None}
}


# ==============================================================================
# FUNÇÕES GEOMÉTRICAS E UTILITÁRIOS
# ==============================================================================
def point_from_box(box):
    """Retorna o centroide de uma bounding box [x1, y1, x2, y2]."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def get_needle_tip(center_point, boxes_by_class, masks_by_class):
    """
    Determina o ponto da ponta da agulha.
    Prioridade 1: Classe 'tip'.
    Prioridade 2: Ponto da máscara de segmentação da 'needle' mais distante do centro.
    """
    if "tip" in boxes_by_class and len(boxes_by_class["tip"]) > 0:
        idx = np.argmax([x[1] for x in boxes_by_class["tip"]])
        return point_from_box(boxes_by_class["tip"][idx][0])

    if "needle" in masks_by_class and len(masks_by_class["needle"]) > 0:
        idx = np.argmax([x[1] for x in boxes_by_class["needle"]])
        mask_points = masks_by_class["needle"][idx]  # Array contendo os contornos (N, 2)
        
        if mask_points is not None and len(mask_points) > 0 and center_point is not None:
            distances = np.linalg.norm(mask_points - center_point, axis=1)
            max_idx = np.argmax(distances)
            return mask_points[max_idx]

    return None


def calculate_value(center, needle_tip, min_pos, max_pos, max_val, min_val=0.0):
    """
    Calcula o valor do manômetro baseado na varredura angular (sentido horário).
    Mapeia a proporção do ângulo da agulha em relação ao arco total [min_pos -> max_pos].
    """
    if any(p is None for p in (center, needle_tip, min_pos, max_pos)) or max_val is None:
        return None

    # Ângulos trigonométricos tradicionais usando coordenadas OpenCV corrigidas (-dy)
    ang_min = np.degrees(np.arctan2(-(min_pos[1] - center[1]), min_pos[0] - center[0])) % 360
    ang_max = np.degrees(np.arctan2(-(max_pos[1] - center[1]), max_pos[0] - center[0])) % 360
    ang_needle = np.degrees(np.arctan2(-(needle_tip[1] - center[1]), needle_tip[0] - center[0])) % 360

    # Varredura horária: a diferença angular decresce no sentido trigonométrico padrão
    sweep_total = (ang_min - ang_max) % 360
    sweep_needle = (ang_min - ang_needle) % 360

    if sweep_total == 0:
        return min_val

    fraction = sweep_needle / sweep_total
    fraction_clamped = max(0.0, min(1.0, fraction))  # Evita flutuações fora da escala física

    return min_val + fraction_clamped * (max_val - min_val)


def parse_ocr_number(text):
    """Filtra strings extraídas pelo OCR isolando apenas valores numéricos flutuantes."""
    if not text:
        return None
    cleaned = text.strip().replace(",", ".")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


# ==============================================================================
# PIPELINE EXECUÇÃO PRINCIPAL
# ==============================================================================
def main(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = YOLO(args.model)

    # Aceita tanto ID de câmera (int) quanto caminhos de arquivos de vídeo (str)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Falha ao abrir a fonte de captura: {source}")

    # Estruturas para armazenar as melhores imagens de calibração obtidas temporariamente
    best_crops = {
        name: {"conf": -1.0, "img": None} 
        for scale in SCALES_CONFIG.keys() 
        for name in [f"{scale}_max", f"{scale}_unit"]
    }

    frame_count = 0
    print("[INFO] Iniciando captura. Analisando os primeiros 100 frames para calibração...")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Fim da transmissão de vídeo ou falha no frame.")
            break

        frame_count += 1
        h, w = frame.shape[:2]
        
        # Predição com o modelo YOLOv8 segmentação
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        boxes_by_class = defaultdict(list)
        masks_by_class = defaultdict(list)

        if results.boxes is not None:
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            has_masks = results.masks is not None

            for i, (cls_id, conf, box) in enumerate(zip(clss, confs, boxes)):
                cname = model.names[cls_id]
                boxes_by_class[cname].append((box, conf))
                if has_masks and i < len(results.masks.xy):
                    masks_by_class[cname].append(results.masks.xy[i])

        # Extração de referências comuns do instrumento
        center_box = max(boxes_by_class["center"], key=lambda x: x[1])[0] if "center" in boxes_by_class else None
        center_pt = point_from_box(center_box) if center_box is not None else None
        needle_pt = get_needle_tip(center_pt, boxes_by_class, masks_by_class)

        # ----------------------------------------------------------------------
        # ETAPA A: CALIBRAÇÃO ATÉ O FRAME 100 (ARMAZENAMENTO DO MELHOR CROP)
        # ----------------------------------------------------------------------
        if frame_count <= CALIBRATION_FRAMES:
            for scale_name, cfg in SCALES_CONFIG.items():
                max_class = f"{cfg['prefix']}max-value"
                unit_class = f"{cfg['prefix']}unit"

                # Atualiza melhor crop para Max Value
                if max_class in boxes_by_class:
                    box, conf = max(boxes_by_class[max_class], key=lambda x: x[1])
                    if conf > best_crops[f"{scale_name}_max"]["conf"]:
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            best_crops[f"{scale_name}_max"] = {"conf": conf, "img": crop.copy()}

                # Atualiza melhor crop para Unit
                if unit_class in boxes_by_class:
                    box, conf = max(boxes_by_class[unit_class], key=lambda x: x[1])
                    if conf > best_crops[f"{scale_name}_unit"]["conf"]:
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            best_crops[f"{scale_name}_unit"] = {"conf": conf, "img": crop.copy()}

            # Exibe status de progresso na janela gráfica
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Calibrando: {frame_count}/{CALIBRATION_FRAMES} frames", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow("Leitor de Manometro Analogico", annotated_frame)
            
            # Condicional disparada EXATAMENTE no frame 100 para executar o OCR
            if frame_count == CALIBRATION_FRAMES:
                print("\n[INFO] Frame 100 atingido. Executando OCR nos melhores frames registrados...")
                reader = get_ocr_reader()

                for scale_name, cfg in SCALES_CONFIG.items():
                    # Processamento do max_value
                    max_data = best_crops[f"{scale_name}_max"]
                    if max_data["img"] is not None:
                        img_path = os.path.join(SAVE_DIR, f"{scale_name}_max_value_best.png")
                        cv2.imwrite(img_path, max_data["img"])
                        
                        ocr_res = reader.readtext(max_data["img"])
                        if ocr_res:
                            # Seleciona o texto com maior confiança do OCR
                            ocr_text = max(ocr_res, key=lambda x: x[2])[1]
                            parsed_val = parse_ocr_number(ocr_text)
                            SCALES_CONFIG[scale_name]["max_val"] = parsed_val
                            print(f" > Escala [{scale_name.upper()}]: Max Detectado = {parsed_val} (Texto: '{ocr_text}')")
                        else:
                            print(f" > Escala [{scale_name.upper()}]: Imagem salva, mas OCR não identificou caracteres.")
                    
                    # Processamento da unidade
                    unit_data = best_crops[f"{scale_name}_unit"]
                    if unit_data["img"] is not None:
                        img_path = os.path.join(SAVE_DIR, f"{scale_name}_unit_best.png")
                        cv2.imwrite(img_path, unit_data["img"])
                        
                        ocr_res = reader.readtext(unit_data["img"])
                        if ocr_res:
                            ocr_text = max(ocr_res, key=lambda x: x[2])[1]
                            SCALES_CONFIG[scale_name]["unit"] = ocr_text.strip()
                            print(f" > Escala [{scale_name.upper()}]: Unidade Detectada = '{ocr_text.strip()}'")

                print(f"[INFO] Calibracao concluida. Imagens de maior confianca salvas na pasta: '{SAVE_DIR}'\n")

        # ----------------------------------------------------------------------
        # ETAPA B: LEITURA GEOMÉTRICA EM TEMPO REAL (APÓS FRAME 100)
        # ----------------------------------------------------------------------
        else:
            display_strings = []

            for scale_name, cfg in SCALES_CONFIG.items():
                # Alvos de busca geométrica fixa
                min_class = f"{cfg['prefix']}min-value"
                max_class = f"{cfg['prefix']}max-value"

                # Atualiza posições físicas (mecanismo de cache caso falhe alguma detecção pontual)
                if min_class in boxes_by_class:
                    cfg["last_min_pt"] = point_from_box(max(boxes_by_class[min_class], key=lambda x: x[1])[0])
                if max_class in boxes_by_class:
                    cfg["last_max_pt"] = point_from_box(max(boxes_by_class[max_class], key=lambda x: x[1])[0])

                # Caso a escala nunca tenha definido um valor máximo válido via OCR, ela é pulada
                if cfg["max_val"] is None:
                    continue

                # Processa o cálculo se houver referências espaciais completas
                calculated_val = calculate_value(
                    center=center_pt,
                    needle_tip=needle_pt,
                    min_pos=cfg["last_min_pt"],
                    max_pos=cfg["last_max_pt"],
                    max_val=cfg["max_val"],
                    min_val=0.0 # Min_value fixado permanentemente em 0
                )

                if calculated_val is not None:
                    display_strings.append(f"{scale_name.upper()}: {calculated_val:.2f} {cfg['unit']}")

            # Renderização de tela customizada
            annotated_frame = results.plot() # Plota máscaras e caixas nativas do YOLO
            
            # Barra de exibição superior preta para contraste do texto
            cv2.rectangle(annotated_frame, (0, 0), (w, 65), (0, 0, 0), -1)
            output_text = " | ".join(display_strings) if display_strings else "Aguardando deteccao completa..."
            
            cv2.putText(annotated_frame, output_text, (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Leitor de Manometro Analogico", annotated_frame)

        # Interrupção manual do loop via teclado
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leitor geometrico de manometros usando mascaras YOLOv8")
    parser.add_argument("--model", type=str, required=True, help="Caminho do arquivo de pesos .pt (YOLOv8-seg)")
    parser.add_argument("--source", type=str, default="0", help="ID da webcam (ex: 0) ou caminho de um arquivo de video")
    args = parser.parse_args()
    main(args)