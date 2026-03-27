from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import csv
import base64
from dbconn import get_collection, get_mongo_client
import math

stop_processing = False

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
        if class_id in [0, 10]:  # Person (0), Helmet (10)
            rois.append([xmin, ymin, xmax, ymax, class_id, score, tracker_id])
    return rois

def process_video_helmet(session_id: str, input_path, threshold: float, mongo_credentials: dict, frame_interval: int = 1):
    global stop_processing
    stop_processing = False

    # Setup folders and files
    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")
    threshold = threshold / 100
    print(threshold)

    # MongoDB setup
    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]
    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["helmet_safety"]

    csv_file = os.path.join(csv_folder, f"helmet_safety_by_frames_{session_id}.csv")
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "Safe", "Unsafe", "Alert Status", "Frame"])

    # Load model and setup video output
    model = YOLO("yolo8n.pt")  # Update with your model path
    output_video = f"output_videos/output_video_{session_id}.mp4"
    os.makedirs("output_videos", exist_ok=True)
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
    unique_tracker_ids = set()
    persistent_status = {}
    centroid_threshold = 110  # Distance threshold for helmet proximity

    while cap.isOpened():
        if stop_processing:
            print("Stopping YOLO process...")
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        results = model.track(frame, persist=True, conf=threshold, iou=0.5, agnostic_nms=True)
        detections = detect_boxes(results)

        person_ppe = {}
        frame_safe = 0
        frame_unsafe = 0

        # First pass: store all bounding boxes and centroids
        for det in detections:
            xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
            if tracker_id == -1:
                continue

            centroid = calculate_centroid(xmin, ymin, xmax, ymax)
            if class_id == 0:  # Person
                person_ppe[tracker_id] = {
                    "helmet": False,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "centroid": centroid
                }
            elif class_id == 10:  # Helmet
                person_ppe[tracker_id] = {"bbox": [xmin, ymin, xmax, ymax], "centroid": centroid} if tracker_id not in person_ppe else person_ppe[tracker_id]

        # Second pass: helmet assignment based on centroid proximity
        for det in detections:
            xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
            if tracker_id == -1 or class_id == 0:
                continue

            if class_id == 10:  # Helmet
                helmet_centroid = calculate_centroid(xmin, ymin, xmax, ymax)
                for tid in person_ppe:
                    if tid != -1 and tid != tracker_id and "centroid" in person_ppe[tid]:
                        person_centroid = person_ppe[tid]["centroid"]
                        distance = math.sqrt((helmet_centroid[0] - person_centroid[0])**2 + 
                                             (helmet_centroid[1] - person_centroid[1])**2)
                        if distance < centroid_threshold:
                            person_ppe[tid]["helmet"] = True
                            persistent_status[tid] = True

        # Third pass: drawing, counting, and status tracking
        for det in detections:
            xmin, ymin, xmax, ymax, class_id, score, tracker_id = det
            if class_id == 0 and tracker_id != -1:
                has_helmet = False

                if tracker_id in persistent_status and persistent_status[tracker_id]:
                    has_helmet = True
                else:
                    has_helmet = person_ppe[tracker_id]["helmet"] if tracker_id in person_ppe else False
                    persistent_status[tracker_id] = has_helmet  # Save status even if unsafe

                # Per-frame safe/unsafe count
                if has_helmet:
                    frame_safe += 1
                else:
                    frame_unsafe += 1

                # Total unique count (once per person)
                if tracker_id not in unique_tracker_ids:
                    if has_helmet:
                        safe_count += 1
                    else:
                        unsafe_count += 1
                    unique_tracker_ids.add(tracker_id)


                label = f"Safe (Helmet)" if has_helmet else "Unsafe (No Helmet)"
                color = (0, 255, 0) if has_helmet else (0, 0, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Updated alert_status logic: check if any unsafe person is still in frame
        current_ids_in_frame = {det[6] for det in detections if det[6] != -1}
        unsafe_in_frame = any(
            (tid in persistent_status and not persistent_status[tid])
            for tid in current_ids_in_frame
        )
        alert_status = unsafe_in_frame

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

    print(safe_count, unsafe_count)
    return safe_count, unsafe_count

def stop_yolo_helmet():
    global stop_processing
    stop_processing = True
