from ultralytics import YOLO
import uuid
import os
import time
import cv2
import csv
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dbconn import get_collection, get_mongo_client
import base64
from io import BytesIO

def is_inside_roi(point, roi_coordinates):
    """ Check if a point (x, y) is inside the polygon ROI """
    return cv2.pointPolygonTest(np.array(roi_coordinates, dtype=np.int32), point, False) >= 0

def process_video_crowd(session_id: str, video_path: str, threshold: float, roi_data: list, mongo_credentials: dict, 
                        frame_interval: int = 1, person_area: int = 30000):  # Changed person_area to 5000
    """
    Process video for crowd monitoring with ROI-based crowd density analysis
    
    Args:
        session_id: Unique identifier for the session
        video_path: Path to the input video
        threshold: Detection confidence threshold (0-1 range)
        roi_data: List of (x, y) coordinates defining the ROI polygon
        mongo_credentials: Dictionary with MongoDB connection details
        frame_interval: Process every nth frame
        person_area: Estimated area occupied by one person (pixels²)
    
    Returns:
        output_video_path: Path to the processed output video
        detection_start_time: Time when first detection occurred
        detection_duration: Total time of detection (in seconds)
    """
    start_time = time.time()
    output_video_path = f"output_videos/output_video_{session_id}.mp4"
    os.makedirs("output_videos", exist_ok=True)
    
    # Load YOLO model - using m version as in code 2
    model = YOLO('yolov8m.pt')
    
    # Alert threshold percentage (for density)
    alert_threshold = threshold  # Match the 60% from code 2
    
    now = datetime.now()

    # MongoDB connection
    try:
        connection_string = mongo_credentials.get("connection_string", "")
        password = mongo_credentials.get("password", "")
        db_name = mongo_credentials.get("db_name", "")
        
        client, database = get_mongo_client(connection_string, password, db_name)
        collection = database["crowd_monitoring"]
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        collection = None

    current_date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, 0
        
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # Video duration in seconds
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Calculate ROI area - make sure roi_data is correctly formatted
    roi_np = np.array(roi_data, dtype=np.int32)
    roi_area = cv2.contourArea(roi_np)
    
    if roi_area <= 0:
        print(f"Error: Invalid ROI area: {roi_area}. Check ROI coordinates: {roi_data}")
        return None, 0
    
    # Setup variables for tracking
    frame_count = 0
    detection_start_time = None
    detection_end_time = None
    first_detection = True
    max_percentage = 0.0
    
    # Create CSV file - keeping your original format
    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)
    csv_file_path = os.path.join(csv_folder, f"crowd_monitoring_{session_id}.csv")

    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "Percentage_in_roi", "Alert_status", "Frame"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Calculate actual video timestamp (independent of processing speed)
        video_timestamp = (frame_count / fps)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Process every nth frame to improve performance
        if frame_count % frame_interval == 0:
            # Run YOLOv8 detection - using conf threshold directly as in code 2
            results = model(frame)
            
            # Count people inside ROI
            people_inside_roi = 0
            try:
                detections = results[0].boxes.data.cpu().numpy()
                
                for det in detections:
                    class_id = int(det[5])
                    
                    # Only count persons (class 0)
                    if class_id == 0:  # Person class
                        x1, y1, x2, y2 = map(int, det[:4])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        if is_inside_roi((cx, cy), roi_data):
                            people_inside_roi += 1
                            # Draw person center point
                            cv2.circle(display_frame, (cx, cy), 4, (255, 0, 0), -1)
                            # Draw bounding box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            except Exception as e:
                print(f"Error processing detections: {e}")
                continue
            
            # Calculate density percentage - using the same logic as code 2
            percentage_in_roi = (people_inside_roi * person_area) / roi_area * 100
            percentage_in_roi = min(percentage_in_roi, 100.0)  # Cap at 100%
            
            # Update max percentage
            max_percentage = max(max_percentage, percentage_in_roi)
            
            # Determine alert status - using the 60% threshold from code 2
            alert_status = percentage_in_roi > alert_threshold
            roi_color = (0, 0, 255) if alert_status else (0, 255, 0)
            
            # For display: HIGH/NORMAL text
            alert_display_text = "HIGH" if alert_status else "NORMAL"
            
            # Track detection times
            if people_inside_roi > 0:
                if first_detection:
                    detection_start_time = video_timestamp
                    first_detection = False
                detection_end_time = video_timestamp
            
            # Draw ROI on the display frame
            cv2.polylines(display_frame, [np.array(roi_data, dtype=np.int32)], 
                         isClosed=True, color=roi_color, thickness=2)
            
            # Add information text to frame, similar to code 2's format
            cv2.putText(display_frame, f"Alert: {alert_display_text} ({percentage_in_roi:.1f}%)", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi_color, 2)
            
            cv2.putText(display_frame, f"People in ROI: {people_inside_roi}", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert frame to base64 for CSV - keeping your original format
            _, buffer = cv2.imencode(".jpg", display_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Write to CSV - with alert_status as true/false
            with open(csv_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_count, 
                    timestamp, 
                    round(percentage_in_roi, 2), 
                    "True" if alert_status else "False",
                    frame_base64
                ])
        
        # Write the updated frame to output video
        out.write(display_frame)
    
    # Calculate detection duration
    detection_duration = 0
    if detection_start_time is not None and detection_end_time is not None:
        detection_duration = detection_end_time - detection_start_time
    
    # If no detection occurred, set start time to 0
    if detection_start_time is None:
        detection_start_time = 0
    
    # Release video objects
    cap.release()
    out.release()
    
    # Record data to MongoDB
    try:
            collection.insert_one({
                "session_id": session_id,
                "date": current_date,
                "time": current_time,
                "max_density_percentage": round(max_percentage, 2),
                "detection_start_time": timestamp,
                "detection_duration": detection_duration,
            })
    except Exception as e:
            print(f"MongoDB error: {e}")
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Detection start time: {detection_start_time:.2f}s")
    print(f"Detection duration: {detection_duration:.2f}s")
    print(f"Maximum density: {max_percentage:.2f}%")
    
    return timestamp, detection_duration