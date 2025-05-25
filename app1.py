from ultralytics import YOLO
import cv2
import numpy as np

# 分別加载不同任务的 YOLOv11 模型
model_detect = YOLO('yolo11s.pt')         # 物體偵測
model_segment = YOLO('yolo11s-seg.pt')    # 實例分割
model_classify = YOLO('yolo11s-cls.pt')   # 圖像分類
model_pose = YOLO('yolo11s-pose.pt')      # 姿態估計
model_obb = YOLO('yolo11s-obb.pt')        # 定向邊界框

# 读取图像
image = cv2.imread('image1.jpg')  # 替換為你的圖像路徑
assert image is not None, "图像读取失败，请检查路径是否正确"

# ----------------------
# 物体检测
# ----------------------
def object_detection(image):
    results = model_detect(image)
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------
# 实例分割
# ----------------------
def instance_segmentation(image):
    results = model_segment(image)
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Instance Segmentation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------
# 图像分类
# ----------------------
def image_classification(image):
    results = model_classify(image)
    for r in results:
        if r.probs is not None:
            cls_id = r.probs.top1
            conf = r.probs.top1conf.item()  # 使用 top1conf 取得信心值
            print(f"Class: {r.names[cls_id]} (confidence: {conf:.2f})")
        else:
            print("未检测到分类结果")

# ----------------------
# 姿态估计
# ----------------------
def pose_estimation(image):
    results = model_pose(image)
    kpts = results[0].keypoints
    if kpts is not None:
        keypoints = kpts.xy.cpu().numpy()
        for person in keypoints:
            for x, y in person:
                cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.imshow("Pose Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------
# 定向边界框 (OBB)
# ----------------------
def oriented_bbox(image):
    results = model_obb(image)
    boxes = results[0].obb
    if boxes is not None and hasattr(boxes, 'xy') and boxes.xy is not None:
        polys = boxes.xy.cpu().numpy()
        for box in polys:
            pts = box.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imshow("Oriented Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------
# 整合所有任务
# ----------------------
def process_image(image):
    print("Running Object Detection...")
    object_detection(image.copy())

    print("Running Instance Segmentation...")
    instance_segmentation(image.copy())

    print("Running Image Classification...")
    image_classification(image.copy())

    print("Running Pose Estimation...")
    pose_estimation(image.copy())

    print("Running Oriented Bounding Boxes (OBB)...")
    oriented_bbox(image.copy())

# ----------------------
# 主流程执行
# ----------------------
process_image(image)
