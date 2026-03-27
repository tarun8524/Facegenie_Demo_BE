# Milk Spillage 
import numpy as np
from datetime import datetime
import cv2
import os
import csv,base64
from dbconn import get_collection, get_mongo_client
stop_processing = False

def process_video2(session_id: str, input_path, mongo_credentials: dict, frame_interval: int = 5): 
    global stop_processing  
    stop_processing = False
    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time1 = now.strftime("%H:%M:%S")
    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]

    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["milk_spillage"]

    csv_file = os.path.join(csv_folder, f"milk_spillage_by_frames_{session_id}.csv")
    # Initialize CSV with headers if it doesn't exist
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "Approx. Wastage Percentage", "Detection Start Time", "Total Detection Time", "Alert Status", "Frame"])

    output_video = f"output_videos_2/output_video2.mp4"

    # Check if the output video already exists, if yes, delete it
    if os.path.exists(output_video):
        os.remove(output_video)
    
    output_frame_dir = "output_frame"
    os.makedirs(output_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    counterA_roi = [[120, 830], [778, 627], [956, 998], [773, 1075], [55, 1075], [23, 914]]
    counterA_roi_poly = np.array(counterA_roi, dtype=np.int32)
    detection_start_time, total_detection_time, is_currently_detecting, alert_status = None, 0, False, False
    WHITE_THRESHOLD = 0.02
    detection_start_time_str = ""

    out = cv2.VideoWriter(output_video, fourcc, fps, (1600, 900))

    if not cap.isOpened():
        print("Failed to load the video. Check the path.")
    else:
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        display_scale = 0.5
        frame_width = int(original_width * display_scale)
        frame_height = int(original_height * display_scale)

        frame_number = 0

        try:
            while True:
                if stop_processing:
                    print("Stopping YOLO process...")
                    break
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed to read frame.")
                    break

                frame_number += 1  # Increment frame count
                timestamp = datetime.now().strftime("%H:%M:%S")  # Get the current time

                frame_resized = cv2.resize(frame, (frame_width, frame_height))
                output_frame = frame_resized.copy()

                hsv = cv2.cvtColor(output_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 25, 255]))

                scale_x, scale_y = frame_width / original_width, frame_height / original_height
                scaled_roi = [[int(x * scale_x), int(y * scale_y)] for x, y in counterA_roi]
                scaled_roi_poly = np.array(scaled_roi, dtype=np.int32)

                roi_mask = np.zeros_like(mask)
                cv2.fillPoly(roi_mask, [scaled_roi_poly], 255)
                masked_white = cv2.bitwise_and(mask, roi_mask)

                roi_area = cv2.countNonZero(roi_mask)
                white_pixels = cv2.countNonZero(masked_white)
                white_percentage = white_pixels / roi_area if roi_area > 0 else 0

                if white_percentage > WHITE_THRESHOLD:
                    if not is_currently_detecting:
                        if detection_start_time is None:
                            detection_start_time = datetime.now()
                        is_currently_detecting = True
                        alert_status = True

                    if detection_start_time is not None:
                        total_detection_time = (datetime.now() - detection_start_time).seconds

                else:
                    if is_currently_detecting:
                        is_currently_detecting = False
                        detection_start_time = None
                    total_detection_time = 0

                detected_white_in_roi = cv2.bitwise_and(output_frame, output_frame, mask=masked_white)
                combined_frame = np.hstack((frame_resized, detected_white_in_roi))
                combined_frame = combined_frame[0:535, 0:1515]
                combined_frame_resized = cv2.resize(combined_frame, (1600, 900))

                try:
                    if detection_start_time is not None:
                        detection_start_time_str = detection_start_time.strftime("%H:%M:%S")
                except:
                    detection_start_time_str = ""

                out.write(combined_frame_resized)
                _, buffer = cv2.imencode(".jpg", frame)  # Use the display_frame with all drawings
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                # **Live Saving Data to CSV - all frames, with frame_base64 conditional**
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if frame_number % 15 == 1:  # Save frame_base64 for frames 1, 16, 31, etc.
                        writer.writerow([frame_number, timestamp, round(white_percentage*100, 2), 
                                       detection_start_time_str, total_detection_time, alert_status, frame_base64])

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

    try:
        collection.insert_one({
            "current_date": current_date,
            "current_Time": current_time1,
            "Wastage_percentage": round(white_percentage*100, 2),
            "Detection_start_time": detection_start_time_str,
            "Total_detection_time": total_detection_time
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")

    return round(white_percentage*100, 2), detection_start_time_str, total_detection_time

# Call this function from another thread to stop YOLO
def stop_yolo2():
    global stop_processing
    stop_processing = True