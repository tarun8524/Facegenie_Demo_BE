from ultralytics import YOLO
import uuid
import os
import time
import cv2
import csv
import math
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from urllib.parse import quote_plus
from dbconn import get_collection, get_mongo_client
import base64
from io import BytesIO

def is_inside_roi(point, roi_coordinates):
    """ Check if a point (x, y) is inside the polygon ROI """
    return cv2.pointPolygonTest(np.array(roi_coordinates, dtype=np.int32), point, False) >= 0

def get_first_frame_base64(stream_url: str) -> str:
    """ Retrieve the first frame from a video stream and return it as base64 """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError("Could not open video stream")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read first frame from stream")
    
    # Convert frame to base64
    _, buffer = cv2.imencode(".jpg", frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return frame_base64

def process_video(session_id: str, stream_url: str, threshold: int, roi_data: list, mongo_credentials: dict, 
                  frame_interval: int = 12, speed_threshold: float = 0.1, idle_threshold: int = 10):
    start_time = time.time()
    output_video_path = f"output_videos/output_video_{session_id}.mp4"
    model = YOLO('yolov8m.pt')
    threshold = threshold / 100
    now = datetime.now()

    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]

    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["product"]

    current_date = now.strftime("%d-%m-%Y")
    current_time1 = now.strftime("%H:%M:%S")

    def detect_boxes(results):
        boxes = results[0].boxes
        rois = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            score = int(box.conf[0] * 100)
            class_id = int(box.cls[0])
            tracker_id = int(box.id[0]) if box.id is not None else None
            rois.append([xmin, ymin, xmax, ymax, class_id, score, tracker_id])
        return rois

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError("Could not open video stream")
        
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    seen_ids = {}
    total_count = 0
    person_count = 0
    total_dwell_time = total_active_time = total_idle_time = 0.0
    persons_id = []
    frame_count = 0

    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)

    csv_file_path = os.path.join(csv_folder, f"person_count_by_frames_{session_id}.csv")

    # Use the requested CSV header format
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no","Timestamp", "Number of Persons", "Total Dwell Time", "Frame"])

    # Track which IDs are present in the current frame
    active_tracker_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        current_time = frame_count / fps  # Use frame-based timing as fallback

        display_frame = frame.copy()
        roi_coordinates_int = np.array(roi_data, dtype=np.int32)
        cv2.polylines(display_frame, [roi_coordinates_int], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(display_frame, "ROI Region", tuple(roi_coordinates_int[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        num_persons = 0
        total_dwell_time_frame = 0.0
        active_tracker_ids.clear()

        if frame_count % frame_interval == 0:
            results = model.track(frame, persist=True, conf=threshold, iou=0.5, agnostic_nms=True)
            rois = detect_boxes(results)
            print(f"Frame {frame_count}: Detected {len(rois)} objects")

            for roi in rois:
                if roi[4] == 0 and roi[5] > 70:  # Lowered confidence threshold
                    tracker_id = roi[6]
                    xmin, ymin, xmax, ymax = roi[:4]
                    curr_x, curr_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                    if is_inside_roi((curr_x, curr_y), roi_data):
                        active_tracker_ids.add(tracker_id)
                        num_persons += 1

                        if tracker_id not in seen_ids:
                            seen_ids[tracker_id] = {
                                'start_time': current_time,
                                'end_time': current_time,
                                'bbox': (xmin, ymin, xmax, ymax),
                                'prev_position': (curr_x, curr_y),
                                'total_distance': 0.0,
                                'active_time': 0.0,
                                'idle_time': 0.0
                            }
                            total_count += 1
                        else:
                            prev_x, prev_y = seen_ids[tracker_id]['prev_position']
                            distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                            seen_ids[tracker_id]['total_distance'] += distance
                            seen_ids[tracker_id]['prev_position'] = (curr_x, curr_y)
                            seen_ids[tracker_id]['end_time'] = current_time
                            seen_ids[tracker_id]['bbox'] = (xmin, ymin, xmax, ymax)

                            time_increment = frame_interval / fps
                            if distance <= idle_threshold:
                                seen_ids[tracker_id]['idle_time'] += time_increment
                            else:
                                seen_ids[tracker_id]['active_time'] += time_increment

                        dwell_time = seen_ids[tracker_id]['end_time'] - seen_ids[tracker_id]['start_time']
                        total_dwell_time_frame += dwell_time

                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        text = f"ID: {tracker_id}, Dwell: {dwell_time:.1f}s"
                        cv2.putText(display_frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            total_dwell_time = sum(data['end_time'] - data['start_time'] for data in seen_ids.values())
            print(f"Frame {frame_count}: {num_persons} persons in ROI, Dwell {total_dwell_time_frame:.2f}s")

        out.write(display_frame)

        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", display_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            with open(csv_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([frame_count, timestamp, num_persons, round(total_dwell_time_frame, 2), frame_base64])

    # Final statistics
    person_count = 0
    total_dwell_time = 0
    total_active_time = 0
    total_idle_time = 0
    for tracker_id, data in seen_ids.items():
        dwell_time = data['end_time'] - data['start_time']
        avg_speed = data['total_distance'] / dwell_time if dwell_time > 0 else 0
        data['active_time'] = max(0, dwell_time - data['idle_time'])  # Prevent negative active time

        total_dwell_time += dwell_time
        total_active_time += data['active_time']
        total_idle_time += data['idle_time']

        if dwell_time > 0.5:  # Lowered threshold
            person_count += 1

    print(f"Final Person Count: {person_count}")
    print(f"Total Dwell Time: {total_dwell_time:.2f} seconds")
    print(f"Total Active Time: {total_active_time:.2f} seconds")
    print(f"Total Idle Time: {total_idle_time:.2f} seconds")

    try:
        collection.insert_one({
            "current_date": current_date,
            "current_Time": current_time1,
            "Person_Count": person_count,
            "Total_Dwell_Time": total_dwell_time,
            "Total_Active_Time": total_active_time,
            "Total_Idle_Time": total_idle_time
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")

    return output_video_path, person_count, total_dwell_time