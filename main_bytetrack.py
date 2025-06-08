import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from tracker_bytetrack.byte_tracker import BYTETracker
from ultralytics.utils import LOGGER
LOGGER.setLevel(50)

CONFIDENCE_THRESHOLD = 0.367

class TrackerArgs:
    def __init__(self,
                 track_thresh=0.6,
                 track_buffer=15,
                 match_thresh=0.5,
                 mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20

    def __str__(self):
        return (f"TrackerArgs(track_thresh={self.track_thresh}, "
                f"track_buffer={self.track_buffer}, "
                f"match_thresh={self.match_thresh}, "
                f"mot20={self.mot20})")

args = TrackerArgs(track_thresh=0.4, track_buffer=15, match_thresh=0.5, mot20=False)
print("Using Tracker Parameters:", args)

tracker = BYTETracker(TrackerArgs(), frame_rate=30)

YOLO_CLASS_MAP = {0: "Car", 1: "Pedestrian"}

KITTI_PATH = "C:/yichi/kitti/training/image_02/"
SEQUENCES = sorted(os.listdir(KITTI_PATH))

model = YOLO("yolov8m-kitti-best-infer.pt", verbose=False)
print(model.overrides)


OUTPUT_PATH = "C:/yichi/PycharmProjects/YOLOv8_Project/TrackEval/data/trackers/kitti/kitti_2d_box_train/T13/data/"
VISUAL_OUTPUT_PATH = "C:/yichi/bt_visual_output1/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_PATH, exist_ok=True)

total_time_all = 0.0
total_frames_all = 0

def iou(box1, box2):
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

for seq_id in SEQUENCES:
    seq_path = os.path.join(KITTI_PATH, seq_id)
    image_files = sorted(os.listdir(seq_path))

    track_results = []
    track_id_to_cls = {}

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(seq_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        img_info = [height, width]
        img_size = (height, width)

        start_time = time.time()

        detections = []
        detection_info = []
        results = model(image)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                cls = int(cls)
                if cls not in YOLO_CLASS_MAP or score < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, score])
                detection_info.append(((x1, y1, x2, y2), float(score), YOLO_CLASS_MAP[cls]))

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{YOLO_CLASS_MAP[cls]} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        detections_np = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(detections_np, img_info, img_size) if len(detections_np) > 0 else []

        end_time = time.time()
        total_time_all += (end_time - start_time)
        total_frames_all += 1

        frame_results = []
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id
            score = t.score

            # IOU 匹配 detection，确定类别与置信度
            best_iou = 0
            best_cls = "Car"
            best_score = 0.0
            for det_box, det_score, det_cls in detection_info:
                current_iou = iou((x1, y1, x2, y2), det_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_cls = det_cls
                    best_score = det_score

            track_id_to_cls[track_id] = best_cls

            frame_results.append([
                i, track_id, best_cls,
                0, 0, 0,
                x1, y1, x2, y2,
                -1, -1, -1,
                -1000, -1000, -1000,
                -10, best_score
            ])

            color = (0, 255, 0) if best_cls == "Car" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"ID {track_id} {best_cls} {best_score:.2f}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        track_results.append(frame_results)

        visual_output_path = os.path.join(VISUAL_OUTPUT_PATH, f"{seq_id}_{image_file}")
        cv2.imwrite(visual_output_path, image)

    output_txt = os.path.join(OUTPUT_PATH, f"{seq_id}.txt")
    with open(output_txt, "w") as f:
        for frame in track_results:
            for t in frame:
                f.write(" ".join(map(str, t)) + "\n")

    print(f"Finished sequence {seq_id}")

fps_all = total_frames_all / total_time_all if total_time_all > 0 else 0
print(f"\nAverage FPS: {fps_all:.2f} ({total_frames_all} frames in {total_time_all:.2f} seconds)")
