# conveyour Belt Crate Count
from ultralytics import YOLO
import cv2
from datetime import datetime
import numpy as np
import matplotlib.path as mplPath
import os,base64
import csv
from dbconn import get_collection, get_mongo_client
stop_processing = False

def detect_box(results):
    boxes = results[0].boxes
    bboxes = boxes.xyxy
    scores = boxes.conf
    classes = boxes.cls
    ids = boxes.id  # Tracker IDs
 
    rois = []
    for index in range(len(boxes)):
        xmin = int(bboxes[index][0])
        ymin = int(bboxes[index][1])
        xmax = int(bboxes[index][2])
        ymax = int(bboxes[index][3])
        score = int(scores[index] * 100)
        class_id = int(classes[index])
        tracker_id = int(ids[index]) if ids is not None else None  # Tracker ID
        rois.append([xmin, ymin, xmax, ymax, class_id, score, tracker_id])
    return rois

def process_video4(session_id: str,input_path,mongo_credentials: dict,frame_interval : int =5):
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
    collection = database["conveyor_belt_crate_count"]

    csv_file = os.path.join(csv_folder, f"conveyor_crate_count_by_frames_{session_id}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no", "Timestamp", "Total Crate Count","Frame"])

    unique_ids_list = []
    box_count = 0

    section1_roi = [[820, 1010], [968, 1152], [1057, 1096], [889, 960]]
    section1_roi_poly = np.array(section1_roi)
    section1_roi_poly_path = mplPath.Path(section1_roi_poly)

    out_video = f"output_videos/output_video_{session_id}.mp4"
    # Check if the output video already exists, if yes, delete it
    if os.path.exists(out_video):
        os.remove(out_video)
    output_frame_dir = "output_frame"
    os.makedirs(output_frame_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, (original_width, original_height))

    model = YOLO('crate_count.pt')

    frame_number = 0

    while cap.isOpened():
        if stop_processing:
                print("Stopping YOLO process...")
                break
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        timestamp = datetime.now().strftime("%H:%M:%S")  # Get current time

        results = model.track(frame, persist=True, conf=0.65, iou=0.5, agnostic_nms=True)
        rois = detect_box(results)
    
        for roi in rois:
            x1, y1, x2, y2, class_id, score, tracker_id = roi  
            person_centroid = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(frame, person_centroid, 5, (0, 0, 255), -1)  # Red circle at the centroid

            if section1_roi_poly_path is not None:  # Check if centroid is inside ROI
                if section1_roi_poly_path.contains_point(person_centroid):
                    if tracker_id not in unique_ids_list:
                        unique_ids_list.append(tracker_id)

            box_count = len(unique_ids_list)
            bbox_message = f"Packet {score}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            
            (text_width, text_height), baseline = cv2.getTextSize(bbox_message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x, text_y = x1, y1 - text_height
            cv2.rectangle(frame, (text_x, text_y), (text_x + text_width, text_y + text_height + baseline), (0, 255, 0), thickness=cv2.FILLED)
            cv2.putText(frame, bbox_message, (text_x, text_y + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        
        cv2.polylines(frame, pts=[section1_roi_poly], isClosed=True, color=(255, 102, 102), thickness=2)  

        out.write(frame)
        _, buffer = cv2.imencode(".jpg", frame)  # Use the display_frame with all drawings
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # **Live Saving Data to CSV**
        with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if frame_number % 15 == 1:
                        writer.writerow([frame_number, timestamp, box_count,frame_base64]) # Ensure data is written immediately

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    try:
        collection.insert_one({
            "current_date": current_date,
            "current_Time": current_time1,
            "Total_crates": box_count
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")

    return box_count


# Call this function from another thread to stop YOLO
def stop_yolo4():
    global stop_processing
    stop_processing = True
