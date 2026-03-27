#Crate Count
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import csv,base64
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

def process_video1(session_id: str,input_path,mongo_credentials: dict,frame_interval: int = 5): 
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
    collection = database["crate_count"]

    csv_file = os.path.join(csv_folder, f"crate_count_by_frames_{session_id}.csv")
    
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no","Timestamp","Crates","Crates_count","Frame"])  

    model = YOLO('crate_detection.pt')  
    output_video = f"output_videos/output_video_{session_id}.mp4"

    if os.path.exists(output_video):
        os.remove(output_video)


    unique_tracker_ids = set()
    tracker_centroids_in_roi = set()
    roi_box_count = 0

    roi_start = (1034, 448)
    roi_end = (1736, 1318)
    
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_number = 0  

    while cap.isOpened():
        if stop_processing:
            print("Stopping YOLO process...")
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1  
        timestamp = datetime.now().strftime("%H:%M:%S")

        results = model.track(frame, persist=True, conf=0.85, iou=0.5, agnostic_nms=True)
        rois = detect_box(results)

        for roi in rois:
            x1, y1, x2, y2, score, tracker_id = roi[0], roi[1], roi[2], roi[3], roi[5], roi[6]  
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2

            if tracker_id is not None and tracker_id not in unique_tracker_ids:
                if (x1 >= roi_start[0] and y1 >= roi_start[1] and x2 <= roi_end[0] and y2 <= roi_end[1]):
                    if (centroid_x, centroid_y) not in tracker_centroids_in_roi:
                        roi_box_count += 1  
                        tracker_centroids_in_roi.add((centroid_x, centroid_y))  
                        unique_tracker_ids.add(tracker_id)  

            bbox_message = f"ID: {tracker_id} {score}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  

            (text_width, text_height), baseline = cv2.getTextSize(bbox_message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x, text_y = x1, y1 - text_height
            cv2.rectangle(frame, (text_x, text_y), (text_x + text_width, text_y + text_height + baseline), (0, 255, 0), thickness=cv2.FILLED)
            #cv2.putText(frame, bbox_message, (text_x, text_y + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        cv2.rectangle(frame, roi_start, roi_end, (255, 0, 255), 2)
        
        crates_count = roi_box_count * 10  

        out.write(frame)

        _, buffer = cv2.imencode(".jpg", frame)  # Use the display_frame with all drawings
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if frame_number % 15 == 1:
                        writer.writerow([frame_number, timestamp, roi_box_count,crates_count, frame_base64])

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    try:
        collection.insert_one({
            "current_date": current_date,
            "current_Time": current_time1,
            "Roi_box_crate": roi_box_count,
            "Total_crates": roi_box_count * 10,
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")

    return roi_box_count, roi_box_count * 10

# Call this function from another thread to stop YOLO
def stop_yolo1():
    global stop_processing
    stop_processing = True

