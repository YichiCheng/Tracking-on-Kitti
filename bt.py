import os
import cv2
import numpy as np
from ultralytics import YOLO
from byte_tracker import BYTETracker
from ultralytics.utils import LOGGER

LOGGER.setLevel(50)  

# define parameters
class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.7  
        self.track_buffer = 15  
        self.match_thresh = 0.5  
        self.mot20 = False  

# COCO → KITTI
COCO_TO_KITTI = {2: "Car", 0: "Pedestrian"}  # Car → "Car", Pedestrian → "Pedestrian"

KITTI_PATH = "C:/yichi/kitti/training/image_02/"
SEQUENCES = sorted(os.listdir(KITTI_PATH))

model = YOLO("yolov8m.pt", verbose=False)

tracker = BYTETracker(TrackerArgs(), frame_rate=30)

OUTPUT_PATH = "C:/yichi/PycharmProjects/YOLOv8_Project/TrackEval/data/trackers/kitti/kitti_2d_box_train/T1/data/"
VISUAL_OUTPUT_PATH = "C:/yichi/bt_visual_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_PATH, exist_ok=True)

for seq_id in SEQUENCES:
    seq_path = os.path.join(KITTI_PATH, seq_id)
    image_files = sorted(os.listdir(seq_path))

    track_results = []
    track_id_to_cls = {}  # track_id → class name

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(seq_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f" Warning: Unable to load {image_path}, skipping...")
            continue
            
        height, width = image.shape[:2]
        img_info = [height, width]
        img_size = (height, width)  # use 640x640

        # launch YOLO （Car n Pedestrian）
        detections = []
        cls_list = []  
        results = model(image, classes=[0, 2])

        print(f"Frame {i}: YOLO detection finished")
        print(f"YOLO Model Config: {model.overrides}") 
        print(f"Original Image Size: {img_info}")
        print(f"YOLO Detected Objects: {len(results[0].boxes)}")

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  
            scores = r.boxes.conf.cpu().numpy()  
            classes = r.boxes.cls.cpu().numpy() 

            for box, score, cls in zip(boxes, scores, classes):
                cls = int(cls)
                if cls not in COCO_TO_KITTI:
                    continue  # skip non-Car/Pedestrian

                x1, y1, x2, y2 = map(int, box)  
                detections.append([x1, y1, x2, y2, score])
                cls_list.append(COCO_TO_KITTI[cls])  # switch to KITTI cls name

                #  YOLO box vs（blue）
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{COCO_TO_KITTI[cls]} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        detections = np.array(detections, dtype=np.float32) if len(detections) > 0 else np.empty((0, 5),
                                                                                                 dtype=np.float32)

        print(f" Frame {i}: {detections.shape[0]} detections found\n{detections}")

        # launch ByteTrack trackiing
        if len(detections) > 0:
            tracks = tracker.update(detections, img_info, img_size)
        else:
            tracks = []

        print(f"Frame {i}: {len(tracks)} objects tracked")

        # save tracking result
        frame_results = []
        for idx, t in enumerate(tracks):
            x1, y1, x2, y2 = map(int, t.tlbr)  
            track_id = t.track_id
            score = t.score
            
            print(f" Track ID {track_id}: TLWH = {t.tlwh}, TLBR = {t.tlbr}")
        
            cls = track_id_to_cls.get(track_id, cls_list[idx] if idx < len(cls_list) else "Car")
            track_id_to_cls[track_id] = cls 

            frame_results.append([
                i, track_id, cls,
                0, 0, 0,
                x1, y1, x2, y2,
                -1, -1, -1,
                -1000, -1000, -1000,
                -10, score
            ])

            #  ByteTrack box（green）**
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

    print(f" Finished sequence {seq_id}, results saved to {output_txt}")
