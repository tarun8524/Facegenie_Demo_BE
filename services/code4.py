from ultralytics import YOLO
import uuid
import os
import time
import cv2
import csv
import numpy as np
from dbconn import get_collection, get_mongo_client
from datetime import datetime
import base64

def calculate_overlap_percentage(bbox, roi_coordinates):
    """
    Calculate what percentage of the bounding box area is inside the ROI.
    
    Args:
        bbox: Tuple of (xmin, ymin, xmax, ymax) - the bounding box coordinates
        roi_coordinates: List of (x, y) tuples forming a polygon ROI
    
    Returns:
        Float between 0 and 1 indicating the percentage overlap
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Create a mask for the ROI
    mask = np.zeros((max(ymax + 10, 1000), max(xmax + 10, 1000)), dtype=np.uint8)
    roi_np = np.array(roi_coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [roi_np], 255)
    
    # Create a mask for the bounding box
    bbox_mask = np.zeros_like(mask)
    cv2.rectangle(bbox_mask, (xmin, ymin), (xmax, ymax), 255, -1)
    
    # Calculate overlap
    intersection = cv2.bitwise_and(mask, bbox_mask)
    intersection_area = cv2.countNonZero(intersection)
    bbox_area = (xmax - xmin) * (ymax - ymin)
    
    if bbox_area == 0:  # Prevent division by zero
        return 0
    
    return intersection_area / bbox_area

def is_inside_roi(point, roi_coordinates):
    """ Check if a point (x, y) is inside the polygon ROI """
    return cv2.pointPolygonTest(np.array(roi_coordinates, dtype=np.int32), point, False) >= 0

def process_video_intrusion(session_id:str, video_path:str, threshold: int, roi_data:list, 
                           mongo_credentials: dict, frame_interval=1, alert_threshold=1, 
                           detection_threshold=0.7):
    """
    Process video for intrusion detection with improved accuracy.
    
    Args:
        session_id: Unique identifier for this processing session
        video_path: Path to the input video file
        threshold: Detection confidence threshold (0-100)
        roi_data: List of (x, y) coordinates defining the ROI polygon
        mongo_credentials: Dictionary with MongoDB connection details
        frame_interval: Process every nth frame
        alert_threshold: Minimum number of persons in ROI to trigger alert
        overlap_threshold: Minimum percentage of bbox that must be in ROI to count as intrusion (0.0-1.0)
    
    Returns:
        Tuple of (intrusion_time_sec, max_persons_in_roi)
    """
    # Create session ID and output paths
    output_video_path = f"output_videos/output_video_{session_id}.mp4"
    os.makedirs("output_videos", exist_ok=True)
    
    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]

    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["Intrusion_detection"]
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time1 = now.strftime("%H:%M:%S")
    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)
    overlap_threshold=threshold/100
    # CSV file path
    csv_file_path = os.path.join(csv_folder, f"intrusion_detection_{session_id}.csv")
    
    # Load YOLO model
    model = YOLO('yolov8m.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0, 0
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize CSV file
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "No_of_persons_in_ROI", "Alert_status", "Frame"])
    
    frame_count = 0
    intrusion_time_sec = 0  # Track time in seconds instead of total intrusions
    time_per_frame = 1 / fps  # Time per frame in seconds
    time_per_processed_frame = time_per_frame * frame_interval  # Time between processed frames
    max_persons_in_roi = 0  # Track maximum number of persons in ROI
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Process only every nth frame based on frame_interval
        if frame_count % frame_interval == 0:
            # Create a copy for display
            display_frame = frame.copy()
            
            # Draw ROI on display frame
            roi_coordinates_int = np.array(roi_data, dtype=np.int32)
            cv2.polylines(display_frame, [roi_coordinates_int], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(display_frame, "Restricted Area", tuple(roi_coordinates_int[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Run detection
            results = model.predict(frame, conf=detection_threshold, classes=[0])  # Only detect people (class 0)
            
            # Count persons in ROI
            persons_in_roi = 0
            if len(results) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Only process person detections
                    if class_id == 0:
                        # Calculate center point of the bounding box
                        center_x = (xmin + xmax) // 2
                        center_y = (ymin + ymax) // 2
                        
                        # Calculate the percentage of the bounding box that overlaps with the ROI
                        overlap_percentage = calculate_overlap_percentage((xmin, ymin, xmax, ymax), roi_data)
                        
                        # Check if the overlap percentage exceeds our threshold
                        if overlap_percentage >= overlap_threshold:
                            persons_in_roi += 1
                            
                            # Draw red bounding box for persons inside ROI
                            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"INTRUDER", (xmin, ymin - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            # Draw yellow bounding box for persons near but not sufficiently inside ROI
                            if is_inside_roi((center_x, center_y), roi_data) or overlap_percentage > 0:
                                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                                cv2.putText(display_frame, f"Near ROI", (xmin, ymin - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Update max persons in ROI if needed
            if persons_in_roi > max_persons_in_roi:
                max_persons_in_roi = persons_in_roi
            
            # Determine alert status
            alert_status = persons_in_roi >= alert_threshold
            
            if alert_status:
                # Increment the intrusion time based on the frame interval
                intrusion_time_sec += time_per_processed_frame
                
                # Add alert text to display frame
                cv2.putText(display_frame, f"ALERT: {persons_in_roi} person(s)", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Display intrusion time on frame
                cv2.putText(display_frame, f"Total intrusion time: {intrusion_time_sec:.2f}s", 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # # Display current overlap threshold
                # cv2.putText(display_frame, f"Overlap threshold: {overlap_threshold:.2f}", 
                #            (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(display_frame)
            
            # Convert frame to base64 for CSV storage
            _, buffer = cv2.imencode(".jpg", display_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Write data to CSV
            with open(csv_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_count,
                    timestamp,
                    persons_in_roi,
                    "True" if alert_status else "False",
                    frame_base64
                ])
        
        # Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Intrusion detection completed:")
    print(f"- Total frames processed: {frame_count}")
    print(f"- Total intrusion time: {intrusion_time_sec:.2f} seconds")
    print(f"- Maximum persons in ROI: {max_persons_in_roi}")

    try:
        collection.insert_one({
            "Current_date": current_date,
            "Current_time": current_time1,
            "Intrusion_time_seconds": round(intrusion_time_sec, 2),
            "Max_persons_in_ROI": max_persons_in_roi
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")
    
    return intrusion_time_sec, max_persons_in_roi