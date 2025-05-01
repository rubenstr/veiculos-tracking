import cv2
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "videos", "video.mp4")
NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CSV_PATH = os.path.join(BASE_DIR, "results", "csv", f"resultado_veiculos_{NOW}.csv")


# print("Caminho absoluto:", BASE_DIR)
# print("Arquivo existe?", os.path.exists(VIDEO_PATH))

# Função para calcular IoU (Interseção sobre União)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# Carregar modelo YOLO mais robusto
model = YOLO("yolov8m.pt")

# Iniciar DeepSORT
deepsort = DeepSort(max_age=30)

# Carregar vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
#cap = cv2.VideoCapture("\\data/videos/ideo.mp4")
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Inicializar contadores por tipo
vehicle_count = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorbike': 0
}

# Para garantir que cada veículo só seja contado uma vez
counted_ids = {
    'car': set(),
    'truck': set(),
    'bus': set(),
    'motorbike': set()
}

# Configuração do CSV
csv_file = CSV_PATH
csv_header = ['Frame', 'ID', 'Tipo', 'Confiança', 'Centro_X', 'Centro_Y']
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

frame_id = 0
track_id_to_class = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    frame_id += 1

    # Detectar objetos com YOLO
    results = model(frame)[0]
    detections = []
    det_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)

        if cls_name in vehicle_count and conf > 0.6:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2, y2], conf, cls_name))
            det_boxes.append((x1, y1, x2, y2, cls_name))

    trackers = deepsort.update_tracks(detections, frame=frame)

    for track in trackers:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_tlbr())
        cx, cy = (l + r) // 2, (t + b) // 2

        if track_id not in track_id_to_class:
            best_iou = 0
            best_cls = "desconhecido"
            for (x1, y1, x2, y2, cls_name) in det_boxes:
                iou = compute_iou((l, t, r, b), (x1, y1, x2, y2))
                if iou > best_iou:
                    best_iou = iou
                    best_cls = cls_name
            track_id_to_class[track_id] = best_cls

        vehicle_class = track_id_to_class[track_id]
        if vehicle_class in vehicle_count:
            if track_id not in counted_ids[vehicle_class]:
                vehicle_count[vehicle_class] += 1
                counted_ids[vehicle_class].add(track_id)

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} | {vehicle_class}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_id, track_id, vehicle_class, "-", cx, cy])

    # Exibir contagem
    y_offset = 30
    for vtype, count in vehicle_count.items():
        text = f"{vtype.capitalize()}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    cv2.imshow("Detecção e Rastreio de Veículos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
