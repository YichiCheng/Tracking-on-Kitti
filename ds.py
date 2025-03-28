import os
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from deep_sort_realtime.deepsort_tracker import DeepSort
LOGGER.setLevel(50)  # 只保留关键错误日志

# 定义 DeepSORT 参数
tracker = DeepSort(
    max_age=5,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=None,
)

# COCO → KITTI 类别字符串映射
COCO_TO_KITTI = {2: "Car", 0: "Pedestrian"}  # Car → "Car", Pedestrian → "Pedestrian"

KITTI_PATH = "C:/yichi/kitti/training/image_02/"
SEQUENCES = sorted(os.listdir(KITTI_PATH))

# 初始化 YOLOv8
model = YOLO("yolov8m.pt", verbose=False)



# 创建输出目录
OUTPUT_PATH = "C:/yichi/PycharmProjects/YOLOv8_Project/TrackEval/data/trackers/kitti/kitti_2d_box_train/T2/data/"
VISUAL_OUTPUT_PATH = "C:/yichi/ds_visual_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_PATH, exist_ok=True)

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
            print(f"⚠️ Warning: Unable to load {image_path}, skipping...")
            continue

        # 获取图像尺寸
        height, width = image.shape[:2]
        img_info = [height, width]
        img_size = (height, width)  # **保持 YOLO 训练使用的 640x640**

        # 运行 YOLO 目标检测（仅检测 Car 和 Pedestrian）
        detections = []
        cls_list = []  # 存储类别字符串
        results = model(image, classes=[0, 2])

        # **调试信息**
        print(f"🟢 Frame {i}: YOLO 预测完成")
        print(f"YOLO Model Config: {model.overrides}")  # 查看 imgsz
        print(f"Original Image Size: {img_info}")
        print(f"YOLO Detected Objects: {len(results[0].boxes)}")

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # 获取边界框 (xyxy 格式)
            scores = r.boxes.conf.cpu().numpy()  # 获取置信度
            classes = r.boxes.cls.cpu().numpy()  # 获取类别索引

            for box, score, cls in zip(boxes, scores, classes):
                cls = int(cls)
                if cls not in COCO_TO_KITTI:
                    continue  # 过滤非 Car/Pedestrian

                x1, y1, x2, y2 = map(int, box)

                detections.append([(x1, y1, x2 - x1, y2 - y1), float(score), COCO_TO_KITTI[cls]])
                print(f"Frame {i} detections: {detections}")

                cls_list.append(COCO_TO_KITTI[cls])  # 转换为 KITTI 类别名称

                # **可视化 YOLO 预测框（蓝色）**
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{COCO_TO_KITTI[cls]} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 运行 ds 目标跟踪
        if len(detections) > 0:
            tracks = tracker.update_tracks(detections, frame=image)

        else:
            tracks = []

        print(f"🔹 Frame {i}: {len(tracks)} objects tracked")

        # 记录跟踪结果
        frame_results = []
        for idx, t in enumerate(tracks):
            x1, y1, w, h = map(int, t.to_tlwh())  # ✅ DeepSORT 期望的格式
            x2, y2 = x1 + w, y1 + h  # 计算右下角坐标
            track_id = t.track_id

            score = t.get_det_conf()# 可能存储置信度的属性
            if score is None:
                score = 0.0  # 设为默认值，避免 NoneType 格式化错误

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

    print(f"✅ Finished sequence {seq_id}, results saved to {output_txt}")
