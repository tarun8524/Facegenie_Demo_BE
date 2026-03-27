import numpy as np
from datetime import datetime
import cv2
import os
import csv
import base64
from skimage.metrics import structural_similarity as ssim
from dbconn import get_collection, get_mongo_client

stop_processing = False

def process_video_tamper(session_id: str, input_path, mongo_credentials: dict, frame_interval: int = 5):
    global stop_processing
    stop_processing = False
    
    # Create csv folder if it doesn't exist
    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)
    
    # Get current date and time
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")
    
    # MongoDB connection
    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]
    
    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["camera_tampering"]
    
    # CSV setup
    csv_file = os.path.join(csv_folder, f"camera_tampering_{session_id}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "Detection_Type", "Alert_Status", "Frame"])
    
    # Output video setup
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    output_video = f"{output_dir}/tamper_detection_{session_id}.mp4"
    if os.path.exists(output_video):
        os.remove(output_video)
    
    # Output frame directory
    output_frame_dir = "output_frame_tamper"
    os.makedirs(output_frame_dir, exist_ok=True)
    
    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Failed to load the video. Check the path.")
        return "Video Load Error", "", 0
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (original_width, original_height))
    
    # Tampering detection thresholds
    # Using exactly the same thresholds as code1
    blur_threshold = 180
    scene_change_threshold = 0.75
    occlusion_threshold = 100
    
    # Persistent tampering state
    tamper_state = {
        "blur": False,
        "scene_change": False,
        "occlusion": False
    }
    
    alert_duration = 100  # number of frames to persist the alert
    alert_counter = {
        "blur": 0,
        "scene_change": 0,
        "occlusion": 0
    }
    
    frame_number = 0
    detection_start_time = None
    detection_type = "None"
    total_detection_time = 0
    detection_start_time_str = ""
    
    # Read first frame for comparison
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        return "Frame Read Error", "", 0
    
    try:
        while not stop_processing:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break
            
            frame_number += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Copy frame for display
            display_frame = frame.copy()
            
            # Debug print for blur detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # print(f"Frame {frame_number}, Blur score: {laplacian_var}")
            
            # Detect blur - EXACTLY as in code1
            blur_flag = laplacian_var < blur_threshold
            if blur_flag:
                tamper_state["blur"] = True
                alert_counter["blur"] = alert_duration
            elif alert_counter["blur"] > 0:
                alert_counter["blur"] -= 1
            else:
                tamper_state["blur"] = False
            
            # Detect scene change - EXACTLY as in code1
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scene_score, _ = ssim(gray1, gray2, full=True)
            scene_change_flag = scene_score < scene_change_threshold
            
            if scene_change_flag:
                tamper_state["scene_change"] = True
                alert_counter["scene_change"] = alert_duration
            elif alert_counter["scene_change"] > 0:
                alert_counter["scene_change"] -= 1
            else:
                tamper_state["scene_change"] = False
            
            # Occlusion detection - EXACTLY as in code1
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_var = np.var(hist)
            occlusion_flag = hist_var < occlusion_threshold
            
            if occlusion_flag:
                tamper_state["occlusion"] = True
                alert_counter["occlusion"] = alert_duration
            elif alert_counter["occlusion"] > 0:
                alert_counter["occlusion"] -= 1
            else:
                tamper_state["occlusion"] = False
            
            # Update detection state
            alert_status = any(tamper_state.values())
            
            # Track detection time
            if alert_status:
                if detection_start_time is None:
                    detection_start_time = datetime.now()
                    detection_start_time_str = detection_start_time.strftime("%H:%M:%S")
                
                # Find which type of tampering is detected
                for t_type, active in tamper_state.items():
                    if active:
                        detection_type = t_type
                        break
                
                total_detection_time = (datetime.now() - detection_start_time).seconds
            else:
                if detection_start_time is not None:
                    detection_start_time = None
                    total_detection_time = 0
            
            # Display tampering alerts on frame - EXACTLY as in code1
            if any(tamper_state.values()):
                y = 50
                for t_type, active in tamper_state.items():
                    if active:
                        msg = f"TAMPERING DETECTED: {t_type.upper()}"
                        # Match the font parameters with code1
                        cv2.putText(display_frame, msg, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 0, 255), 2)
                        y += 30
            
            # Write to output video
            out.write(display_frame)
            
            # Optional: Save individual frames for debugging
            # if any(tamper_state.values()):
            #     cv2.imwrite(f"{output_frame_dir}/frame_{frame_number}.jpg", display_frame)
            
            # Save frame as base64 for CSV
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Write to CSV (only certain frames to save space)
            if frame_number % frame_interval == 1:
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_number, 
                        timestamp, 
                        detection_type if alert_status else "None", 
                        alert_status, 
                        frame_base64
                    ])
            
            # Update previous frame for next comparison
            prev_frame = frame.copy()
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Processing Error", "", 0
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Store results in MongoDB
    try:
        collection.insert_one({
            "session_id": session_id,
            "current_date": current_date,
            "current_time": current_time,
            "detection_type": detection_type,
            "detection_start_time": detection_start_time_str,
            "total_detection_time": total_detection_time
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")
    
    return detection_type, detection_start_time_str, total_detection_time

def stop_tamper_detection():
    """Call this function from another thread to stop processing."""
    global stop_processing
    stop_processing = True