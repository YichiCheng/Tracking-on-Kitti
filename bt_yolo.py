import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from byte_tracker import BYTETracker
from ultralytics.utils import LOGGER

LOGGER.setLevel(50)  # 只保留关键错误日志

class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.4
        self.track_buffer = 15  # 轨迹缓冲区
        self.match_thresh = 0.5  # 目标匹配的 IOU 阈值
        self.mot20 = False

YOLO_CLASS_MAP = {0: "Car", 1: "Pedestrian"}

KITTI_PATH = "C:/yichi/kitti/training/image_02/"
SEQUENCES = sorted(os.listdir(KITTI_PATH))

# 初始化 YOLOv8
model = YOLO("yolov8m-kitti-best-infer.pt", verbose=False)

# 初始化 ByteTrack
tracker = BYTETracker(TrackerArgs(), frame_rate=30)

# 创建输出目录
OUTPUT_PATH = "C:/yichi/PycharmProjects/YOLOv8_Project/TrackEval/data/trackers/kitti/kitti_2d_box_train/T4/data/"
VISUAL_OUTPUT_PATH = "C:/yichi/bt_visual_output3/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_PATH, exist_ok=True)

# 总时间与总帧数统计（用于整体 FPS）
total_time_all = 0.0
total_frames_all = 0

# 遍历所有 sequence
for seq_id in SEQUENCES:
    seq_path = os.path.join(KITTI_PATH, seq_id)
    image_files = sorted(os.listdir(seq_path))

    track_results = []
    track_id_to_cls = {}  # 记录 track_id → 类别名称

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(seq_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  Warning: Unable to load {image_path}, skipping...")
            continue

        # 获取图像尺寸
        height, width = image.shape[:2]
        img_info = [height, width]
        img_size = (height, width)  # **保持 YOLO 训练使用的 640x640**

        # 记录开始时间（检测+跟踪）
        start_time = time.time()

        # 运行 YOLO 目标检测（仅检测 Car 和 Pedestrian）
        detections = []
        cls_list = []  # 存储类别字符串
        #results = model(image)  # 不要加 classes=[0, 2]，否则无法输出 Pedestrian（class 1）
        results = model(image, imgsz=640)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # 获取边界框 (xyxy 格式)
            scores = r.boxes.conf.cpu().numpy()  # 获取置信度
            classes = r.boxes.cls.cpu().numpy()  # 获取类别索引

            for box, score, cls in zip(boxes, scores, classes):
                cls = int(cls)
                if cls not in YOLO_CLASS_MAP:
                    continue  # 过滤非 Car/Pedestrian

                x1, y1, x2, y2 = map(int, box)  # 取整
                detections.append([x1, y1, x2, y2, score])
                cls_list.append(YOLO_CLASS_MAP[cls])

                # **可视化 YOLO 预测框（蓝色）**
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{YOLO_CLASS_MAP[cls]} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 确保 detections 格式正确
        detections = np.array(detections, dtype=np.float32) if len(detections) > 0 else np.empty((0, 5),
                                                                                                 dtype=np.float32)

        # 运行 ByteTrack 目标跟踪
        if len(detections) > 0:
            tracks = tracker.update(detections, img_info, img_size)
        else:
            tracks = []

        # 记录结束时间并累计
        end_time = time.time()
        frame_time = end_time - start_time
        total_time_all += frame_time
        total_frames_all += 1

        # 记录跟踪结果
        frame_results = []
        for idx, t in enumerate(tracks):
            x1, y1, x2, y2 = map(int, t.tlbr)  # **这里可能是问题所在**
            track_id = t.track_id
            score = t.score

            # 获取类别名称，默认设为 "Car"
            cls = track_id_to_cls.get(track_id, cls_list[idx] if idx < len(cls_list) else "Car")
            track_id_to_cls[track_id] = cls  # 记录类别

            frame_results.append([
                i, track_id, cls,
                0, 0, 0,
                x1, y1, x2, y2,
                -1, -1, -1,
                -1000, -1000, -1000,
                -10, score
            ])

            # **可视化 ByteTrack 跟踪框（绿色/红色）**
            color = (0, 255, 0) if cls == "Car" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"ID {track_id} {cls} {score:.2f}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        track_results.append(frame_results)

        # **保存可视化结果**
        visual_output_path = os.path.join(VISUAL_OUTPUT_PATH, f"{seq_id}_{image_file}")
        cv2.imwrite(visual_output_path, image)

    # **保存 Tracking 结果**
    output_txt = os.path.join(OUTPUT_PATH, f"{seq_id}.txt")
    with open(output_txt, "w") as f:
        for frame in track_results:
            for t in frame:
                f.write(" ".join(map(str, t)) + "\n")

    print(f" Finished sequence {seq_id}")

# 输出模型整体平均 FPS
fps_all = total_frames_all / total_time_all if total_time_all > 0 else 0
print(f"\nAverage FPS: {fps_all:.2f} ({total_frames_all} frames in {total_time_all:.2f} seconds)")
