import cv2
import torch
import numpy as np
import time, datetime
import argparse
from scipy.spatial.distance import cosine
from PIL import Image
import torchvision.transforms.functional as F
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dbconn import get_collection, get_mongo_client
import pandas as pd
import os, csv
import base64

# GPU configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

class FeatureExtractor:
    def __init__(self):
        from torchvision.models import resnet50, ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device).eval()
        self.transform = F.to_tensor

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return np.zeros(2048)
        roi = cv2.cvtColor(cv2.resize(roi, (224, 224)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            features = self.model(self.transform(roi).unsqueeze(0).to(device)).squeeze().cpu().detach().numpy()
        return features

class LineSpeedEstimator:
    def __init__(self, line1, line2, distance_meters, fps=30):
        self.lines = [line1, line2]
        self.distance = distance_meters
        self.crossings = {}
        self.speeds = {}
        self.fps = fps  # Use exact fps from video for timing calculations
        self.calibration_factor = 2.5  # Default calibration factor
        self.history_length = 5  # Number of speed measurements to keep for smoothing
        self.min_crossing_frames = 1  # Minimum frames between line crossings to be valid
        self.completed_tracks = set()  # Tracks that have crossed both lines
        self.track_completion_times = {}  # Store when tracks completed crossing both lines
        self.roi_spacing = 32  # Spacing between ROI lines in pixels
        
    def set_calibration(self, factor):
        """Set calibration factor to adjust for real-world conditions"""
        self.calibration_factor = factor

    def check_crossing(self, track_id, centroid, timestamp, frame_number):
        """
        Check if a vehicle has crossed lines and calculate speed
        Uses frame_number for timing instead of system time for hardware independence
        """
        direction = None
        
        # Initialize track data if not exists
        if track_id not in self.crossings:
            self.crossings[track_id] = {
                'signs': [],
                'frame_numbers': {},  # Store frame numbers instead of times
                'last_speeds': []
            }
        
        # Check each line for crossing
        for i, line in enumerate(self.lines):
            (x1, y1), (x2, y2) = line
            
            # Calculate line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            
            # Determine which side of the line the centroid is on
            sign = a*centroid[0] + b*centroid[1] + c
            
            # Store the sign for this track
            if len(self.crossings[track_id]['signs']) <= i:
                self.crossings[track_id]['signs'].append(sign)
            
            # Check if sign has changed (line crossing occurred)
            if len(self.crossings[track_id]['signs']) > i and np.sign(sign) != np.sign(self.crossings[track_id]['signs'][i]):
                line_key = f'line{i+1}'
                
                # Only record crossing if we haven't seen this line before or enough frames have passed
                if (line_key not in self.crossings[track_id]['frame_numbers'] or 
                    abs(frame_number - self.crossings[track_id]['frame_numbers'].get(line_key, 0)) > self.min_crossing_frames):
                    
                    # Store the crossing frame number
                    self.crossings[track_id]['frame_numbers'][line_key] = frame_number
                    
                    # If we have crossed both lines, calculate speed
                    if len(self.crossings[track_id]['frame_numbers']) == 2:
                        line_keys = sorted(self.crossings[track_id]['frame_numbers'].keys())
                        direction = '→' if line_keys[0] == 'line1' else '←'
                        
                        # Calculate frame difference between line crossings
                        f1, f2 = [self.crossings[track_id]['frame_numbers'][k] for k in line_keys]
                        frame_diff = abs(f2 - f1)
                        
                        # Convert frame difference to time in seconds using video FPS
                        # This ensures speed calculation is independent of processing power
                        time_diff = frame_diff / self.fps
                        
                        # Avoid division by zero
                        if time_diff > 0.001:
                            # Calculate speed (km/h) with calibration factor
                            speed = (self.distance / time_diff) * 3.6 * self.calibration_factor
                            
                            # Store speed in history for smoothing
                            self.crossings[track_id]['last_speeds'].append(speed)
                            if len(self.crossings[track_id]['last_speeds']) > self.history_length:
                                self.crossings[track_id]['last_speeds'] = self.crossings[track_id]['last_speeds'][-self.history_length:]
                            
                            # Use median of recent speed measurements for more stability
                            median_speed = np.median(self.crossings[track_id]['last_speeds'])
                            self.speeds[track_id] = (median_speed, direction)
                            
                            # Mark this track as completed
                            self.completed_tracks.add(track_id)
                            # Store the completion time using frame number
                            self.track_completion_times[track_id] = frame_number
                
                # Update the sign for this line
                self.crossings[track_id]['signs'][i] = sign
            
        return self.speeds.get(track_id, (0, None))
        
    def should_display_track(self, track_id, current_frame, display_duration_frames=60):
        """
        Determine if a track should still be displayed after crossing both lines
        Uses frame numbers instead of time for hardware independence
        """
        if track_id not in self.completed_tracks:
            return True  # Always display tracks that haven't crossed both lines
            
        # Check if the track completed crossing within the display duration (in frames)
        completion_frame = self.track_completion_times.get(track_id, 0)
        return (current_frame - completion_frame) <= display_duration_frames

class VehicleTracker:
    def __init__(self, model_type='yolov8'):
        self.model_type = model_type
        self.model = self._load_model()
        self.tracks = {}
        self.next_id = 1
        self.fe = FeatureExtractor()
        self.conf_threshold = 0.6  # Confidence threshold for detections
        self.iou_threshold = 0.3  # IoU threshold for tracking
        self.max_age = 30  # Frames a track can exist without being matched
        self.min_hits = 3  # Minimum hits before a track is displayed
        self.min_box_area = 1000  # Minimum bounding box area
        self.max_box_area = 150000  # Maximum bounding box area
        self.track_history = {}  # Store history of tracks
        
    def _load_model(self):
        if self.model_type == 'faster_rcnn':
            model = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
        else:
            model = YOLO('yolov8x.pt').to(device)
        return model

    def detect_vehicles(self, frame):
        if self.model_type == 'faster_rcnn':
            tensor = F.to_tensor(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            with torch.no_grad():
                dets = self.model(tensor)[0]
            boxes = [d[:4].tolist() for d, s, l in zip(dets['boxes'], dets['scores'], dets['labels']) 
                    if l in [2, 3, 4, 6, 7, 8] and s > self.conf_threshold]
        else:
            results = self.model(frame, verbose=False)[0]
            boxes = [box.xyxy[0].tolist() for box in results.boxes 
                    if int(box.cls) in [2, 3, 5, 7] and box.conf > self.conf_threshold]
        
        # Filter boxes by area
        valid_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if self.min_box_area < area < self.max_box_area:
                valid_boxes.append(box)
        
        # Apply non-maximum suppression
        if len(valid_boxes) > 0:
            valid_boxes = self._apply_nms(valid_boxes, 0.4)
        return valid_boxes
    
    def _apply_nms(self, boxes, threshold):
        """Apply non-maximum suppression to reduce multiple detections per vehicle"""
        if len(boxes) == 0:
            return []
            
        # Convert to numpy for easier manipulation
        boxes = np.array(boxes)
        
        # Calculate areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by area
        order = np.argsort(areas)[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
            
        return boxes[keep].tolist()

    def update_tracks(self, frame, boxes, frame_number):
        """
        Update tracks with new detections
        Added frame_number parameter for hardware-independent tracking
        """
        # Handle the case when there are no existing tracks
        if len(self.tracks) == 0:
            for box in boxes:
                self.tracks[self.next_id] = {
                    'bbox': box,
                    'centroids': [self._get_centroid(box)],
                    'age': 0,
                    'hits': 1,
                    'frame_seen': frame_number,  # Use frame number instead of time
                    'velocity': [0, 0],
                    'disappeared_count': 0
                }
                self.next_id += 1
            return
        
        # Update age for all tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
        
        # Handle the case when there are no detections
        if len(boxes) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['disappeared_count'] += 1
                if self.tracks[tid]['age'] > self.max_age or self.tracks[tid]['disappeared_count'] > 10:
                    del self.tracks[tid]
            return
        
        # Calculate IoU between all detections and existing tracks
        iou_matrix = np.zeros((len(boxes), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for i, box in enumerate(boxes):
            for j, tid in enumerate(track_ids):
                iou_matrix[i, j] = self._iou(box, self.tracks[tid]['bbox'])
        
        # Match tracks to detections based on IoU
        matched_indices = []
        
        # First, try to match with higher IoU threshold
        for i in range(len(boxes)):
            max_iou = self.iou_threshold
            match_tid = None
            
            for j, tid in enumerate(track_ids):
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    match_tid = tid
            
            if match_tid is not None:
                # Update the matched track
                old_bbox = self.tracks[match_tid]['bbox']
                new_bbox = boxes[i]
                old_centroid = self._get_centroid(old_bbox)
                new_centroid = self._get_centroid(new_bbox)
                
                # Calculate velocity (movement between frames)
                dx = new_centroid[0] - old_centroid[0]
                dy = new_centroid[1] - old_centroid[1]
                
                # Update track data
                self.tracks[match_tid]['bbox'] = new_bbox
                self.tracks[match_tid]['centroids'].append(new_centroid)
                if len(self.tracks[match_tid]['centroids']) > 10:
                    self.tracks[match_tid]['centroids'] = self.tracks[match_tid]['centroids'][-10:]
                self.tracks[match_tid]['age'] = 0
                self.tracks[match_tid]['hits'] += 1
                self.tracks[match_tid]['frame_seen'] = frame_number  # Use frame number instead of time
                self.tracks[match_tid]['velocity'] = [dx, dy]
                self.tracks[match_tid]['disappeared_count'] = 0
                matched_indices.append(i)
        
        # Create new tracks for unmatched detections
        for i, box in enumerate(boxes):
            if i not in matched_indices:
                # Verify this is likely a new valid object before creating a track
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                # Additional validation for new tracks
                if self.min_box_area < area < self.max_box_area:
                    self.tracks[self.next_id] = {
                        'bbox': box,
                        'centroids': [self._get_centroid(box)],
                        'age': 0,
                        'hits': 1,
                        'frame_seen': frame_number,  # Use frame number instead of time
                        'velocity': [0, 0],
                        'disappeared_count': 0
                    }
                    self.next_id += 1
        
        # Don't remove tracks too quickly after they've disappeared
        for tid in list(self.tracks.keys()):
            # Use a higher age threshold for vehicles that have already been detected multiple times
            max_allowed_age = self.max_age
            if self.tracks[tid]['hits'] > 10:  # Vehicles with established tracks
                max_allowed_age = self.max_age * 2  # Give more frames before removing
            
            # Increment disappeared count if track has no matching detection
            if tid not in [track_ids[j] for i, j in enumerate(range(len(track_ids))) if i in matched_indices]:
                self.tracks[tid]['disappeared_count'] += 1
            
            # Remove if too old or disappeared for too many consecutive frames
            if (self.tracks[tid]['age'] > max_allowed_age or 
                self.tracks[tid]['disappeared_count'] > 15):  # If disappeared for 15 consecutive frames
                del self.tracks[tid]

    def _get_centroid(self, box):
        """Calculate centroid of a bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
        return inter / union if union else 0

def process_video_speed(session_id: str, video_path: str, threshold: int, mongo_credentials: dict, frame_interval: int = 1):
    """
    Process video and calculate vehicle speeds independent of computational power
    Modified to use frame numbers throughout for timing-critical operations
    """
    output_video = f"output_videos/output_video_{session_id}.mp4"
    os.makedirs("output_videos", exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width,height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fixed parameters
    model_type = 'yolov8'
    distance = 10.0  # Default distance in meters
    roi_position = 0.5  # Vertical position of ROI (center of frame)
    roi_height = 200  # Fixed height of ROI
    roi_width = 1250  # Hardcoded width as in the main function
    roi_tilt = 0  # No tilt for ROI
    roi_color = (0, 0, 255)  # Red color for ROI
    roi_thickness = 1  # Line thickness
    
    # Define rectangular ROI
    roi_y1 = int(height * roi_position)
    roi_y2 = roi_y1 + roi_height
    
    # Calculate center of the frame for ROI positioning
    center_x = width // 2
    center_y = (roi_y1 + roi_y2) // 2
    roi_x1 = center_x - roi_width // 2
    roi_x2 = center_x + roi_width // 2
    
    # Calculate the four corners of the rectangle
    rect_points = np.array([
        [roi_x1, roi_y1],  # Top-left
        [roi_x2, roi_y1],  # Top-right
        [roi_x2, roi_y2],  # Bottom-right
        [roi_x1, roi_y2]   # Bottom-left
    ], dtype=np.float32)
    
    # Apply rotation around the center of the rectangle
    angle_rad = np.radians(roi_tilt)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), roi_tilt, 1.0)
    rotated_rect = cv2.transform(np.array([rect_points]), rotation_matrix)[0].astype(np.int32)
    
    # Extract the top and bottom lines for speed calculation
    roi1 = ((rotated_rect[0][0], rotated_rect[0][1]), (rotated_rect[1][0], rotated_rect[1][1]))  # Top line
    roi2 = ((rotated_rect[3][0], rotated_rect[3][1]), (rotated_rect[2][0], rotated_rect[2][1]))  # Bottom line
    
    # Initialize tracker and speed estimator
    tracker = VehicleTracker(model_type)
    speed_estimator = LineSpeedEstimator(roi1, roi2, distance, fps=fps)
    speed_estimator.roi_spacing = roi_height
    csv_folder = "csv_files"
    os.makedirs(csv_folder, exist_ok=True)
    
    # Dictionary to store speed data
    speed_data = []
    
    # Process video
    frame_count = 0  # Initialize frame counter
    processed_count = 0  # Counter for processed frames
    
    # MongoDB connection setup using the provided credentials
    now = datetime.datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    current_time1 = now.strftime("%H:%M:%S")

    connection_string = mongo_credentials["connection_string"]
    password = mongo_credentials["password"]
    db_name = mongo_credentials["db_name"]
    client, database = get_mongo_client(connection_string, password, db_name)
    collection = database["Speed_Monitoring"]
    
    # Create and setup CSV file
    csv_file = os.path.join(csv_folder, f"speed_monitoring_by_frames_{session_id}.csv")
    
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_no","Timestamp","Normal","Overspeed","Frame"])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every nth frame based on frame_interval
        if frame_count % frame_interval != 0:
            frame_count += 1
            continue
        
        # Calculate timestamp based on frame number and fps
        # This ensures timing is independent of computational power
        frame_timestamp = frame_count / fps
        formatted_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Get new vehicle detections for this frame
        boxes = tracker.detect_vehicles(frame)
        
        # Update all tracks with new detections - pass frame number for timing
        tracker.update_tracks(frame, boxes, frame_count)
        
        # Visualization
        vis = frame.copy()
        
        # Draw rotated rectangular ROI
        cv2.polylines(vis, [rotated_rect], True, roi_color, roi_thickness)
        
        # Draw the lines used for speed calculation
        cv2.line(vis, roi1[0], roi1[1], roi_color, roi_thickness+1)
        cv2.line(vis, roi2[0], roi2[1], roi_color, roi_thickness+1)
        
        # Add ROI labels with adjusted positions based on tilt
        label_offset_x = 10 * np.cos(angle_rad)
        label_offset_y = 10 * np.sin(angle_rad)
        
        entry_label_pos = (int(rotated_rect[0][0] + label_offset_x), int(rotated_rect[0][1] - 10 + label_offset_y))
        exit_label_pos = (int(rotated_rect[3][0] + label_offset_x), int(rotated_rect[3][1] - 10 + label_offset_y))
        
        # Change labels from Entry/Exit Line to ROI Region
        cv2.putText(vis, "ROI Region", entry_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        cv2.putText(vis, "ROI Region", exit_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        
        # Draw distance label in the middle of the ROI
        mid_x = int((rotated_rect[0][0] + rotated_rect[2][0]) / 2)
        mid_y = int((rotated_rect[0][1] + rotated_rect[2][1]) / 2)
        cv2.putText(vis, f"{distance}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        
        # Initialize frame data with counts for each speed category
        frame_data = {
            'frame_no': frame_count,
            'timestamp': formatted_time,
            'session_id': session_id,
            'normal': 0,
            'medium': 0,
            'overspeed': 0
        }
        
        # Draw vehicles and speeds
        frame_counts = {
            'Normal Speed': 0,
            'Medium Speed': 0,
            'Over Speed': 0
        }
        
        for tid, data in tracker.tracks.items():
            # Only display tracks that have been detected multiple times
            # and haven't disappeared for too many frames
            if data['hits'] >= tracker.min_hits and data['disappeared_count'] < 10:
                x1, y1, x2, y2 = map(int, data['bbox'])
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                
                # Skip boxes that are too small or too large (likely false detections)
                box_area = (x2-x1)*(y2-y1)
                if box_area < tracker.min_box_area or box_area > tracker.max_box_area:
                    continue
                
                # Get speed and direction with frame number for better timing
                # Use frame count for timing calculations - key improvement for hardware independence
                speed, direction = speed_estimator.check_crossing(tid, (cx, cy), frame_timestamp, frame_count)
                
                # Only hide the bounding box if the vehicle has crossed both lines AND is no longer visible
                if (tid in speed_estimator.completed_tracks and 
                    data['disappeared_count'] > 5 and
                    not speed_estimator.should_display_track(tid, frame_count, display_duration_frames=fps*2)):  # 2 seconds worth of frames
                    continue  # Skip drawing this track
                
                # Determine speed category and color based on speed and threshold
                if speed != 0:
                    if speed > threshold:
                        speed_category = "Over Speed"
                        box_color = (0, 0, 255)  # Red in BGR
                    elif speed >= threshold * 0.75:
                        speed_category = "Medium Speed"
                        box_color = (0, 165, 255)  # Orange in BGR
                    else:
                        speed_category = "Normal Speed"
                        box_color = (0, 255, 0)  # Green in BGR
                    
                    # Update frame counts
                    frame_counts[speed_category] += 1
                    
                    # Add vehicle data to speed_data if it has a valid speed
                    if tid in speed_estimator.completed_tracks and tid not in [item.get('track_id') for item in speed_data]:
                        vehicle_data = {
                            'track_id': tid,
                            'speed_kmh': int(speed),
                            'direction': direction,
                            'category': speed_category,
                            'timestamp': formatted_time,
                            'frame_no': frame_count,
                            'session_id': session_id
                        }
                        
                        # Add to local data collection
                        speed_data.append(vehicle_data)
                else:
                    speed_category = ""
                    box_color = (0, 255, 255)  # Default Yellow remains
                
                # Draw bounding box with appropriate color
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw ID and centroid
                cv2.circle(vis, (cx, cy), 4, box_color, -1)
                
                # Display info with black text on colored background
                info_text = f"ID: {tid}"
                if speed != 0:
                    # Round to integer for cleaner display
                    info_text += f" {int(speed)}km/h {speed_category}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw background rectangle with appropriate color
                cv2.rectangle(
                    vis, 
                    (x1, y1-text_height-baseline-5), 
                    (x1+text_width+5, y1-5), 
                    box_color, 
                    -1  # Filled rectangle
                )
                
                # Draw text in black
                cv2.putText(
                    vis, 
                    info_text, 
                    (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 0),  # Black color
                    2
                )
        
        # Show GPU/CPU info in the corner
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        cv2.putText(vis, f"Running on: {device_info}", (10, height-20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show progress in the corner
        progress_text = f"Progress: {frame_count}/{total_frames} ({int(100*frame_count/total_frames)}%)"
        cv2.putText(vis, progress_text, (10, height-50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update frame data with counts from this frame
        frame_data['normal'] = frame_counts['Normal Speed'] 
        frame_data['medium'] = frame_counts['Medium Speed']
        frame_data['overspeed'] = frame_counts['Over Speed']
        
        # Encode the current frame as base64 for storage
        _,buffer = cv2.imencode('.jpg', vis)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Write current frame data to CSV - THIS IS THE MISSING PART
        with open(csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_count,
                formatted_time,
                frame_counts['Normal Speed'] + frame_counts['Medium Speed'],
                frame_counts['Over Speed'],
                frame_base64
            ])
        
        # Write processed frame to output video
        out.write(vis)
        
        processed_count += 1
        frame_count += 1
        
        # Print progress periodically
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({int(100*frame_count/total_frames)}%)")
    
    # Clean up
    cap.release()
    out.release()
    
    normal_count = sum(1 for item in speed_data if item.get('category') == 'Normal Speed')
    medium_count = sum(1 for item in speed_data if item.get('category') == 'Medium Speed')
    overspeed_count = sum(1 for item in speed_data if item.get('category') == 'Over Speed')
    print(normal_count,medium_count,overspeed_count)
    
    # Write MongoDB entry with summary data
    try:
        collection.insert_one({
            "current_date": current_date,
            "current_Time": current_time1,
            "normal_count": normal_count + medium_count,
            "overspeed_count": overspeed_count
        })
    except Exception as e:
        print(f"Error: {e} (MongoDB might be disconnected)")
    
    return normal_count+medium_count,  overspeed_count