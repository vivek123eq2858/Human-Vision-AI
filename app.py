from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
from datetime import datetime
import csv
import os
import time
import numpy as np

# ------------------- OCR Setup ------------------- #
# Enhanced OCR with angle classification for better text detection on angled products
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ------------------- CSV Setup ------------------- #
os.makedirs("output", exist_ok=True)
csv_file = "output/detections.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Timestamp", "Object_ID", "Class_Name", "Detected_Text", "Confidence", "Text_Size"])

# ------------------- OCR Extraction with Size Filtering and Keyword Filtering ------------------- #
def extract_large_text(image, min_height_ratio=0.15, min_confidence=0.5, keywords=None):
    """
    Extract only large/prominent text from image, with optional keyword filtering
    
    Args:
        image: Input image crop
        min_height_ratio: Minimum text height as ratio of image height (default: 0.15 = 15%)
        min_confidence: Minimum OCR confidence threshold
        keywords: List of keywords to filter text (e.g., ['BRAND', 'PRODUCT']). If None, no filtering.
    
    Returns:
        List of (text, confidence, text_height) tuples for large text only
    """
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return []
        
        results = ocr.ocr(image, cls=True)  # Use ocr.ocr for better results with angle cls
        
        detections = []
        image_height = image.shape[0]
        
        all_text_data = []
        
        # First pass: collect all text with their sizes
        for result in results:
            if result is None:
                continue
            for line in result:
                if line and len(line) > 1:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text, conf = line[1][0], float(line[1][1])
                    
                    # Skip low confidence
                    if conf < min_confidence:
                        continue
                    
                    # Calculate text bounding box height
                    points = np.array(bbox)
                    y_coords = points[:, 1]
                    text_height = max(y_coords) - min(y_coords)
                    
                    # Calculate text height ratio relative to image
                    height_ratio = text_height / image_height
                    
                    all_text_data.append({
                        'text': text,
                        'conf': conf,
                        'height': text_height,
                        'height_ratio': height_ratio,
                        'bbox': bbox
                    })
        
        if not all_text_data:
            return []
        
        # Find the largest text height in this image
        max_height = max([t['height'] for t in all_text_data])
        
        # Filter: Keep only text that is either:
        # 1. Above the minimum height ratio threshold, OR
        # 2. At least 60% of the largest text height in the image
        for text_data in all_text_data:
            is_large_enough = text_data['height_ratio'] >= min_height_ratio
            is_prominent = text_data['height'] >= (max_height * 0.6)
            
            if is_large_enough or is_prominent:
                # Optional keyword filtering
                if keywords:
                    if any(keyword.upper() in text_data['text'].upper() for keyword in keywords):
                        detections.append((
                            text_data['text'], 
                            text_data['conf'],
                            int(text_data['height'])
                        ))
                else:
                    detections.append((
                        text_data['text'], 
                        text_data['conf'],
                        int(text_data['height'])
                    ))
        
        return detections
        
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return []

# ------------------- Visualization ------------------- #
def draw_results(frame, tracked_objects, class_names):
    """Draw bounding boxes and large text results for all tracked objects"""
    for obj_id, obj_data in tracked_objects.items():
        x1, y1, x2, y2 = obj_data['bbox']
        class_name = obj_data['class']
        texts = obj_data['texts']
        
        # Draw bounding box with thicker line for better visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare label with object ID and class
        id_label = f"ID:{obj_id} [{class_name}]"
        cv2.putText(frame, id_label, (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        
        # Draw large text results with highlight
        if texts:
            # Show top 2 largest texts
            text_label = " | ".join([f"{t[0]}" for t in texts[:2]])
            
            # Create highlighted background for text
            (tw, th), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            
            # Draw yellow highlight background
            cv2.rectangle(frame, (x1-2, y1 - th - 12), (x1 + tw + 8, y1-2), (0, 215, 255), -1)
            # Draw black border
            cv2.rectangle(frame, (x1-2, y1 - th - 12), (x1 + tw + 8, y1-2), (0, 0, 0), 2)
            
            # Draw text in black for contrast
            cv2.putText(frame, text_label, (x1 + 3, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Draw confidence for first text
            conf_label = f"Conf: {texts[0][1]:.2f}"
            cv2.putText(frame, conf_label, (x2 - 100, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    return frame

# ------------------- Main Function ------------------- #
def run_multi_object_ocr(video_path, save_output=False, display_width=1280, ocr_interval=15, 
                         min_text_height_ratio=0.15, min_confidence=0.5, keywords=None, 
                         model_path="yolov8n.pt", save_crops=False):
    """
    Run multi-object detection with large text OCR only, optimized for product detection
    
    Args:
        video_path: Path to video file
        save_output: Whether to save annotated video
        display_width: Width for display window (height auto-calculated to maintain aspect ratio)
        ocr_interval: Perform OCR every N frames (increased for speed)
        min_text_height_ratio: Minimum text height as ratio of crop height (0.15 = 15%)
        min_confidence: Minimum OCR confidence (0.5 = 50%)
        keywords: List of keywords to filter OCR text (e.g., ['BRAND', 'PRODUCT'])
        model_path: Path to YOLO model (use custom trained model for product-specific detection)
        save_crops: Whether to save cropped images of detected objects
    """
    # Load custom or default YOLO model
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened(): 
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Calculate display scale to fit window
    display_scale = display_width / original_width
    display_height = int(original_height * display_scale)
    
    # No delay for maximum speed
    delay = 1

    # Setup video writer
    video_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter("output/annotated_output.mp4", fourcc, fps, 
                                       (original_width, original_height))

    # Dictionary to store all tracked objects
    tracked_objects = {}
    frame_count = 0
    processed_objects = set()

    print("[INFO] Starting multi-object LARGE TEXT OCR detection...")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Display resolution: {display_width}x{display_height}")
    print(f"[INFO] OCR interval: every {ocr_interval} frames (optimized for speed)")
    print(f"[INFO] Min text height ratio: {min_text_height_ratio:.1%}")
    print(f"[INFO] Min confidence: {min_confidence:.1%}")
    if keywords:
        print(f"[INFO] Keyword filtering: {keywords}")
    if save_crops:
        print("[INFO] Saving cropped images to output/crops/")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        start_time = time.time()
        frame_count += 1

        # Run YOLO detection with tracking
        results = model.track(frame, persist=True, verbose=False)
        
        current_frame_objects = {}
        
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get object ID and class
                if hasattr(box, 'id') and box.id is not None:
                    obj_id = int(box.id[0])
                else:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class name
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Store current frame object data
                current_frame_objects[obj_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'texts': []
                }
                
                # Initialize tracked object if new
                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'texts': []
                    }
                else:
                    # Update bbox and class
                    tracked_objects[obj_id]['bbox'] = (x1, y1, x2, y2)
                    tracked_objects[obj_id]['class'] = class_name
                
                # Perform OCR at specified intervals (LARGE TEXT ONLY)
                if frame_count % ocr_interval == 0 and obj_id not in processed_objects:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                        # Save crop if enabled
                        if save_crops:
                            os.makedirs("output/crops", exist_ok=True)
                            cv2.imwrite(f"output/crops/crop_{obj_id}_{frame_count}.jpg", crop)
                        
                        # Extract only large/prominent text with keyword filtering
                        text_detections = extract_large_text(
                            crop, 
                            min_height_ratio=min_text_height_ratio,
                            min_confidence=min_confidence,
                            keywords=keywords
                        )
                        
                        if text_detections:
                            # Sort by text height (largest first)
                            text_detections.sort(key=lambda x: x[2], reverse=True)
                            
                            # Update tracked texts
                            tracked_objects[obj_id]['texts'] = text_detections
                            current_frame_objects[obj_id]['texts'] = text_detections
                            
                            # Log to CSV
                            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                for t, c, h in text_detections:
                                    writer.writerow([
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        obj_id,
                                        class_name,
                                        t,
                                        c,
                                        h
                                    ])
                            
                            processed_objects.add(obj_id)
                    else:
                        # Use previously detected text if crop is too small
                        if obj_id in tracked_objects:
                            current_frame_objects[obj_id]['texts'] = tracked_objects[obj_id]['texts']
                else:
                    # Use previously detected text
                    if obj_id in tracked_objects:
                        current_frame_objects[obj_id]['texts'] = tracked_objects[obj_id]['texts']
        
        # Reset processed objects every interval
        if frame_count % ocr_interval == 0:
            processed_objects.clear()
        
        # Draw results on frame
        frame = draw_results(frame, current_frame_objects, model.names)
        
        # Calculate and display FPS
        fps_val = 1 / (time.time() - start_time + 1e-5)
        
        # Enhanced info display
        info_bg = frame.copy()
        cv2.rectangle(info_bg, (10, 10), (450, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, info_bg, 0.3, 0)
        
        cv2.putText(frame, f"FPS: {fps_val:.1f} | Objects: {len(current_frame_objects)} | Model: {os.path.basename(model_path)}", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Large Text Mode | Frame: {frame_count} | Keywords: {keywords if keywords else 'None'}", 
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize for display to fit laptop screen
        display_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("Multi-Object Large Text OCR", display_frame)

        # Save video if enabled (save at original resolution)
        if save_output and video_writer: 
            video_writer.write(frame)
        
        # Exit on 'q' press
        if cv2.waitKey(delay) & 0xFF == ord('q'): 
            break

    # Cleanup
    cap.release()
    if video_writer: 
        video_writer.release()
    cv2.destroyAllWindows()
    
    print("\n[INFO] Detection complete ✅")
    print(f"[INFO] Total unique objects tracked: {len(tracked_objects)}")
    print(f"[INFO] Results saved to: {csv_file}")
    if save_crops:
        print(f"[INFO] Cropped images saved to: output/crops/")

# ------------------- Run ------------------- #
if __name__ == "__main__":
    video_path = r"C:\Projects\Human Vision AI\video\video12.mp4"
    
    # Run with LARGE TEXT detection only - OPTIMIZED FOR SPEED and PRODUCT DETECTION
    run_multi_object_ocr(
        video_path, 
        save_output=True,
        display_width=400,          # Set width to fit laptop (1280x720 or adjust as needed)
        ocr_interval=15,             # OCR every 15 frames (3x faster than before)
        min_text_height_ratio=0.15,  # Text must be at least 15% of crop height
        min_confidence=0.5,          # Minimum 50% confidence
        keywords=['BRAND', 'PRODUCT', 'NAME'],  # Example: Filter for brand/product-related text (customize or set to None)
        model_path="yolov8n.pt",     # Replace with custom trained model path (e.g., "runs/detect/train/weights/best.pt")
        save_crops=True              # Save cropped images for review
    )