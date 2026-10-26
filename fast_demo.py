import cv2
import numpy as np
import imageio
from ultralytics import YOLO
import os

VIDEO_PATH = "sample_data/sample_highway.mp4"
if not os.path.exists(VIDEO_PATH) and os.path.exists("sample_highway.mp4"):
    VIDEO_PATH = "sample_highway.mp4"

OUTPUT_PATH = "demo_output.mp4"
MAX_FRAMES = 150

model = YOLO("yolov8n-seg.pt")

if os.path.exists("homography_matrix.npy"):
    H = np.load("homography_matrix.npy")
    K = np.load("camera_matrix.npy")
else:
    print("⚠️ .npy files not found. Using fallback matrices for demo.")
    H = np.eye(3) * 50 
    K = np.array([[750, 0, 320], [0, 750, 180], [0, 0, 1]])

fy = K[1, 1]

def image_to_ground(u, v):
    p = np.array([u, v, 1.0])
    world = H @ p
    if abs(world[2]) < 1e-10: return (0.0, 0.0)
    return (world[0]/world[2], world[1]/world[2])

def homography_depth(u, v):
    X, Y = image_to_ground(u, v)
    depth = np.sqrt(X**2 + Y**2)
    return float(depth) if 2.0 < depth < 300.0 else 20.0

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}. Check the file path.")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = imageio.get_writer(OUTPUT_PATH, fps=15, codec='libx264', quality=8, macro_block_size=None)

frame_count = 0
print(f"Processing {MAX_FRAMES} frames on CPU...")

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret: break
    
    results = model.track(frame, persist=True, imgsz=480, verbose=False, classes=[2, 5, 7])
    annotated = frame.copy()
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        tids = results[0].boxes.id.cpu().numpy().astype(int)
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
        
        for i, (box, tid) in enumerate(zip(boxes, tids)):
            x1, y1, x2, y2 = map(int, box)
            center_u = (x1 + x2) / 2.0
            depth = homography_depth(center_u, y2)
            
            mask_h_px = y2 - y1
            height_m = (mask_h_px * depth) / fy
            
            Xl, Yl = image_to_ground(x1, y2)
            Xr, Yr = image_to_ground(x2, y2)
            width_m = np.sqrt((Xr - Xl)**2 + (Yr - Yl)**2)
            
            if 1.0 < width_m < 5.0 and 1.0 < height_m < 6.0: 
                label = f"ID {tid} | W:{width_m:.2f}m H:{height_m:.2f}m"
                color = (0, 255, 0) if width_m < 2.2 else (0, 0, 255)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, max(y1-th-6, 0)), (x1+tw+6, max(y1-12, 0)), (0, 0, 0), -1)
                cv2.putText(annotated, label, (x1, max(y1-12, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if len(masks) > 0:
                    mask_resized = cv2.resize(masks[i], (w, h))
                    mask_bool = mask_resized > 0.5
                    annotated[mask_bool] = annotated[mask_bool] * 0.5 + np.array(color) * 0.5

    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    writer.append_data(rgb_frame)
    frame_count += 1
    
    if frame_count % 25 == 0:
        print(f"Processed {frame_count}/{MAX_FRAMES} frames...")

cap.release()
writer.close()
print(f"✅ Done! Video saved to {os.path.abspath(OUTPUT_PATH)}")
