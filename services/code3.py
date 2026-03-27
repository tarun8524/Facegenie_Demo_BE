from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import csv
import base64
from dbconn import get_collection, get_mongo_client
import math

stop_processing = False

def is_wearing_white(cropped_img):
    # Convert to LAB color space (better for brightness detection)
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    _, bright_mask = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    # Convert to HSV to isolate white color
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    color_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine both masks
    final_mask = cv2.bitwise_and(color_mask, bright_mask)

    white_pixels = cv2.countNonZero(final_mask)
    total_pixels = cropped_img.shape[0] * cropped_img.shape[1]
    
    white_ratio = white_pixels / total_pixels
    return white_ratio > 0.15

def calculate_centroid(xmin, ymin, xmax, ymax):
    return ((xmin + xmax) // 2, (ymin + ymax) // 2)

def detect_boxes(results):
    boxes = results[0].boxes
    rois = []
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        score = float(box.conf[0])
        class_id = int(box.cls[0])
        tracker_id = int(box.id[0]) if box.id is not None else -1
        rois.append([xmin, ymin, xmax, ymax, class_id, score, tracker_id])
    return rois

def process_video_ppe(session_id: str, input_path, threshold: float, mongo_credentials: dict, frame_interval: int = 5):
    global stop_processing
    stop_processing = False
    
    # Setup folders and files
    csv_folder = "csv_files"
    output_folder = "output_videos"
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")
    threshold = threshold / 100
    print(f"Using confidence threshold: {threshold}")
    
    centroid_threshold = 100  # Distance threshold for centroid proximity

    # MongoDB setup
    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]
    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["safety_detection"]

    csv_file = os.path.join(csv_folder, f"safety_detection_by_frames_{session_id}.csv")
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "Safe", "Unsafe", "Alert Status", "Frame"])

    # Load model and setup video output
    model = YOLO("yolov8m.pt")
    output_video = f"{output_folder}/safety_output_{session_id}.mp4"
    if os.path.exists(output_video):
        os.remove(output_video)

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    safe_count = 0
    unsafe_count = 0
    persistent_status = {}  # Track persistent safety status
    unique_tracker_ids = set()  # Track unique person IDs

    while cap.isOpened():
        if stop_processing:
            print("Stopping YOLO process...")
            break
            
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Use tracking with YOLOv8
        results = model.track(frame, persist=True, conf=threshold, iou=0.5, agnostic_nms=True)
        detections = detect_boxes(results)

        person_data = {}
        white_objects = []
        frame_safe = 0
        frame_unsafe = 0

        # First pass: Collect person and potential white object data
        for det in detections:
            xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
            if tracker_id == -1:
                continue

            centroid = calculate_centroid(xmin, ymin, xmax, ymax)
            
            if model.names[class_id] == 'person':
                cropped = frame[ymin:ymax, xmin:xmax]
                has_white = is_wearing_white(cropped)
                person_data[tracker_id] = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "centroid": centroid,
                    "white_detected": has_white,
                    "near_white": False,
                    "score": score
                }
            else:
                # Check if non-person object might be white (potential safety item)
                cropped = frame[ymin:ymax, xmin:xmax]
                if is_wearing_white(cropped):
                    white_objects.append({"centroid": centroid, "tracker_id": tracker_id})

        # Second pass: Check proximity of white objects to persons
        for tracker_id in person_data:
            person_centroid = person_data[tracker_id]["centroid"]
            
            for white_obj in white_objects:
                distance = math.sqrt(
                    (white_obj["centroid"][0] - person_centroid[0])**2 +
                    (white_obj["centroid"][1] - person_centroid[1])**2
                )
                if distance < centroid_threshold:
                    person_data[tracker_id]["near_white"] = True
                    # Update persistent status if white object is close
                    persistent_status[tracker_id] = True
                    break

        # Draw and count
        for tracker_id in person_data:
            xmin, ymin, xmax, ymax = person_data[tracker_id]["bbox"]
            score = person_data[tracker_id]["score"]
            
            # Determine safety status
            if tracker_id in persistent_status and persistent_status[tracker_id]:
                is_safe = True
            else:
                is_safe = person_data[tracker_id]["white_detected"] or person_data[tracker_id]["near_white"]
            
            # Only count each person once in the overall statistics
            if tracker_id not in unique_tracker_ids:
                if is_safe:
                    safe_count += 1
                else:
                    unsafe_count += 1
                unique_tracker_ids.add(tracker_id)
            
            # Always count for the current frame
            if is_safe:
                frame_safe += 1
            else:
                frame_unsafe += 1

            color = (0, 255, 0) if is_safe else (0, 0, 255)
            label = f"{'Safe(PPE)' if is_safe else 'Unsafe(NO PPE)'}"
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        alert_status = frame_unsafe > 0
        out.write(frame)

        # Save frame to CSV every frame_interval
        if frame_number % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([frame_number, timestamp, frame_safe, frame_unsafe, alert_status, frame_base64])

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save to MongoDB
    try:
        collection.insert_one({
            "session_id": session_id,
            "current_date": current_date,
            "current_time": current_time,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")
    print(f"Safety analysis complete - Safe: {safe_count}, Unsafe: {unsafe_count}")
    return safe_count, unsafe_count

def stop_yolo_safety():
    global stop_processing
    stop_processing = True