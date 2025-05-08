import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from tracker_botsort.bot_sort import BoTSORT

LOGGER.setLevel(50)

CONFIDENCE_THRESHOLD = 0.367
YOLO_CLASS_MAP = {0: "Car", 1: "Pedestrian"}

KITTI_PATH = "C:/yichi/kitti/training/image_02/"
SEQUENCES = sorted(os.listdir(KITTI_PATH))

model = YOLO("yolov8m-kitti-best-infer.pt", verbose=False)

class BoTSORTArgs:
    def __init__(self):
        self.track_high_thresh = 0.6
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.5
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.with_reid = False
        self.fast_reid_config = "configs/default.yaml"
        self.fast_reid_weights = "weights/model.pth"
        self.device = "cuda"
        self.cmc_method = "sparseOptFlow"
        self.name = "BoT-SORT"
        self.ablation = ""
        self.mot20 = False

args = BoTSORTArgs()
tracker = BoTSORT(args, frame_rate=30)

OUTPUT_PATH = "C:/yichi/PycharmProjects/YOLOv8_Project/TrackEval/data/trackers/kitti/kitti_2d_box_train/T6/data/"
VISUAL_OUTPUT_PATH = "C:/yichi/botsort_visual_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_PATH, exist_ok=True)

total_time_all = 0.0
total_frames_all = 0

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
        cls_list = []
        results = model(image)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                cls = int(cls)
                print(f"[DEBUG] YOLO Output → Score: {score:.3f}, Class ID: {cls}, Box: {box}")
                if cls not in YOLO_CLASS_MAP or score < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, score, cls])
                cls_list.append(YOLO_CLASS_MAP[cls])

                print(f"[Frame {i}] Detection: Box=({x1},{y1},{x2},{y2}), Score={score:.2f}, Class={YOLO_CLASS_MAP[cls]}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{YOLO_CLASS_MAP[cls]} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if len(detections) > 0:
            detections = np.array(detections, dtype=np.float32)
            print(f"[Frame {i}] Total valid detections: {len(detections)}")
            for d in detections:
                print(f"    → Sent to tracker: {d}")
        else:
            detections = np.empty((0, 6), dtype=np.float32)

        if len(detections) > 0:
            tracks = tracker.update(detections, image)
            print(f"[Frame {i}] Tracker returned {len(tracks)} track(s)")
            if len(tracks) == 0:
                print(f"[DEBUG] [Frame {i}] Tracker failed to associate or create new tracks.")
        else:
            tracks = []

        end_time = time.time()
        frame_time = end_time - start_time
        print(f"[Frame {i}] Processing time: {frame_time:.3f}s")
        total_time_all += frame_time
        total_frames_all += 1

        frame_results = []
        if len(tracks) == 0:
            print(f"[Frame {i}] No tracks returned — drawing detections only")
            #以下这段导致了墨绿色错误可视化框
            for d in detections:
                x1, y1, x2, y2, score, cls_id = map(int, d)
                cls_name = YOLO_CLASS_MAP.get(cls_id, "Unknown")
                cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 0), 2)
                cv2.putText(image, f"{cls_name} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2)

        for idx, t in enumerate(tracks):
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id
            score = t.score
            cls_guess = cls_list[idx] if idx < len(cls_list) else "Car"
            cls = track_id_to_cls.get(track_id, cls_guess)
            track_id_to_cls[track_id] = cls

            print(f"[Frame {i}] Track ID {track_id} → Class: {cls}, Score: {score:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

            frame_results.append([
                i, track_id, cls,
                0, 0, 0,
                x1, y1, x2, y2,
                -1, -1, -1,
                -1000, -1000, -1000,
                -10, score
            ])

            color = (0, 255, 0) if cls == "Car" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"ID {track_id} {cls} {score:.2f}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        track_results.append(frame_results)
        visual_output_path = os.path.join(VISUAL_OUTPUT_PATH, f"{seq_id}_{image_file}")
        cv2.imwrite(visual_output_path, image)

    output_txt = os.path.join(OUTPUT_PATH, f"{seq_id}.txt")
    with open(output_txt, "w") as f:
        for frame in track_results:
            for t in frame:
                f.write(" ".join(map(str, t)) + "\n")

    print(f"[Seq {seq_id}] Finished. Total frames: {len(image_files)}")

fps_all = total_frames_all / total_time_all if total_time_all > 0 else 0
print(f"\n[Summary] Average FPS: {fps_all:.2f} ({total_frames_all} frames in {total_time_all:.2f} seconds)")

