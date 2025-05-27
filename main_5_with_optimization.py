
import cv2
import numpy as np
from ultralytics import YOLO
# import torchreid
import random
import torch
import threading
import time
import random
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from backed_api import fetch_product_data, get_userId, check_out, load_product_data
from object_detection import process_frame
from socket_connection import connect_to_server, send_pickup_event
from datetime import datetime
from helping_function import *
from weight_sensor import *
import socket
from collections import defaultdict
import redis
import multiprocessing
import os
import sys


multiprocessing.set_executable(sys.executable)
if os.name != 'nt':  # For Linux/Unix
    multiprocessing.set_start_method('spawn', force=True)

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# connect with redis
r_decision = redis.Redis(host='127.0.0.1', port=6379, db=0)
r_products = redis.Redis(host='127.0.0.1', port=6379, db=1)

print("Redis connected.........")

############################### Initial defines ###############################
picked_up_objects = {}
json_file = "picked_up_objects.json"
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        picked_up_objects = json.load(f)


transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Function to extract embeddings directly from an image (or OpenCV frame)
def extract_features_from_frame_numpy(ort_session, frame, bbox, transform=transform):
    # model.eval()
    x1, y1, x2, y2 = bbox
    person_crop = frame[y1:y2, x1:x2]
    image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    # Apply transformations and convert image to numpy format
    image = transform(image).unsqueeze(0).numpy()
    # ONNX model input
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    # Run the model to get the embedding
    ort_outs = ort_session.run(None, ort_inputs)
    # Convert output (NumPy) to a PyTorch tensor
    embedding = torch.tensor(ort_outs[0]).squeeze()
    return embedding

def generate_id():
    return random.randint(1000, 9999)


def normalize_embeddings(embedding):
    # L2 normalize the embedding
    return embedding / embedding.norm(p=2, dim=0, keepdim=True)


def extend_wrist_to_palm(elbow, wrist, extension_distance=20):
    # Convert points to numpy arrays for easy vector calculations
    elbow = np.array(elbow)
    wrist = np.array(wrist)
    # Calculate the direction vector from elbow to wrist
    direction_vector = wrist - elbow
    # Normalize the direction vector to a unit vector
    norm = np.linalg.norm(direction_vector)
    unit_vector = direction_vector / norm if norm != 0 else np.array([0, 0])

    # Calculate the extended point by adding the scaled unit vector to the wrist
    extended_point = wrist + unit_vector * extension_distance
    # Return the extended point as a tuple
    return tuple(extended_point)


############################### Entry Camera ###############################

def entry_camera(camera_path, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, entry_area, frame_size):
    
    track_history = {}
    check_out_id_local = ""
    last_update_time = 0
    frame_count = 0
    model = YOLO(yolo_model_name)
    reid_model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    
    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # frame_count += 1
        # # Skip every other frame
        # if frame_count % 2 != 0:
        #     continue

        latest_current_time = time.time()
        frame = cv2.resize(frame, frame_size)
        results = model.track(frame, persist=True, stream=True)
        results = next(results)

        boxes = results.boxes.data
        if len(boxes) != 0:
            for box in boxes:
                x, y, w, h, pose_track_id, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(int(box[4])), box[5]

                person_cx = (x + w) // 2
                person_cy = (y + h) // 2

                # Extract features for the detected person

                features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                features = normalize_embeddings(features)

                with check_out_id_lock:
                    if check_out_id_local != check_out_id.value and check_out_id.value != "":
                        track_history = {k: v for k, v in track_history.items() if v != check_out_id.value}
                        check_out_id_local = check_out_id.value

                # Check if the person is in the entry area
                if entry_area and cv2.pointPolygonTest(np.array(entry_area, np.int32), (person_cx, person_cy), False) >= 0 and conf > 0.8:
                    if pose_track_id not in track_history:
                        random_id = get_userId()
                        # random_id = 'dddade74-868d-4cc6-82fe-c5b516107ec9'
                        # random_id = generate_id()
                        track_history[pose_track_id] = random_id
                        with features_history_lock:
                            features_his[random_id] = features.unsqueeze(0)  # Store initial embedding

                        with global_track_count_lock:
                            if random_id in global_track_count:
                                global_track_count[random_id] += 1
                            else:
                                global_track_count[random_id] = 1


                # If the person is new (not in track_history) and confidence is high
                if pose_track_id not in track_history and conf > 0.7:

                    # features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                    # features = normalize_embeddings(features)

                    best_match_id = None
                    best_similarity = 0

                    with features_history_lock:         
                        for id, stored_embeddings in features_his.items():
                            # Compare the current features with stored embeddings
                            similarity_scores = torch.matmul(stored_embeddings, features)
                            # max_similarity = similarity_scores.max().item()
                            max_similarity = similarity_scores.mean()

                            # Keep track of the best match and ensure the similarity is above the threshold
                            if max_similarity > best_similarity and max_similarity >= similarity_threshold:
                                best_similarity = max_similarity
                                best_match_id = id

                        # If a match is found, update the track history and add new features
                        if best_match_id:
                            track_history[pose_track_id] = best_match_id
                            features_his[best_match_id] = torch.cat([features_his[best_match_id], features.unsqueeze(0)], dim=0)

                            # Keep only the last three embeddings
                            if features_his[best_match_id].size(0) > last_embedding_count:
                                features_his[best_match_id] = features_his[best_match_id][-last_embedding_count:]

                # Continues tracking
                if pose_track_id in track_history:
                    id = track_history[pose_track_id]
                    if conf > 0.7:
                        # if latest_current_time - last_update_time >= 0.2:
                            # features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                            # features = normalize_embeddings(features)
                        with features_history_lock:
                            features_his[id] = torch.cat([features_his[id], features.unsqueeze(0)], dim=0)
                            if features_his[id].size(0) > last_embedding_count:
                                features_his[id] = features_his[id][-last_embedding_count:]
                            # last_update_time = latest_current_time

                    # Draw bounding box and ID
                    cv2.circle(frame, (person_cx, person_cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
                    cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(frame, f"FPS-Count: {np.round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(entry_area, np.int32)], True, (255, 0, 0), 2)
        cv2.imshow(f"YOLOv8 Tracking - {camera_path}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


############################### Exit Camera ################################

def exit_camera(camera_path, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, exit_area, frame_size):

    track_history = {}
    last_update_time = 0
    frame_count = 0
    model = YOLO(yolo_model_name)
    reid_model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # frame_count += 1
        # # Skip every other frame
        # if frame_count % 2 != 0:
        #     continue

        latest_current_time = time.time()
        frame = cv2.resize(frame, frame_size)
        results = model.track(frame, persist=True, stream=True)
        results = next(results)
        
        boxes = results.boxes.data
        if len(boxes) != 0:
            for box in boxes:
                x, y, w, h, pose_track_id, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(int(box[4])), box[5]

                person_cx = (x + w) // 2
                person_cy = (y + h) // 2

                # Extract features for the detected person

                features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                features = normalize_embeddings(features)

                # First time Enter
                if exit_area and cv2.pointPolygonTest(np.array(exit_area, np.int32), (person_cx, person_cy), False) >= 0:
                    if pose_track_id in track_history:
                        id = track_history[pose_track_id]

                        with global_track_count_lock:
                            if id in global_track_count:
                                global_track_count[id] -= 1

                            if global_track_count[id] == 0:

                                if id not in picked_up_objects:
                                    picked_up_objects[id] = {}

                                picked_up_objects[id]["Person checkout"] = "True"
                                save_to_json(picked_up_objects, json_file)
                                out = check_out(id)
                                with check_out_id_lock:
                                    check_out_id.value = id
                                with features_history_lock:
                                    del features_his[id]
                                track_history = {k: v for k, v in track_history.items() if v != id}
                                global_track_count.pop(id)
                                
                #  If the person is new (not in track_history) and confidence is high
                if pose_track_id not in track_history and conf > 0.7:

                    # features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                    # features = normalize_embeddings(features)

                    best_match_id = None
                    best_similarity = 0

                    with features_history_lock:         
                        for id, stored_embeddings in features_his.items():
                            # Compare the current features with stored embeddings
                            similarity_scores = torch.matmul(stored_embeddings, features)
                            # max_similarity = similarity_scores.max().item()
                            max_similarity = similarity_scores.mean()

                            # Keep track of the best match and ensure the similarity is above the threshold
                            if max_similarity > best_similarity and max_similarity >= similarity_threshold:
                                best_similarity = max_similarity
                                best_match_id = id

                        # If a match is found, update the track history and add new features
                        if best_match_id:
                            track_history[pose_track_id] = best_match_id
                            features_his[best_match_id] = torch.cat([features_his[best_match_id], features.unsqueeze(0)], dim=0)

                            # Keep only the last three embeddings
                            if features_his[best_match_id].size(0) > last_embedding_count:
                                features_his[best_match_id] = features_his[best_match_id][-last_embedding_count:]


                # Continues tracking
                if pose_track_id in track_history:
                    id = track_history[pose_track_id]
                    if conf > 0.7:
                        # if latest_current_time - last_update_time >= 1:
                        # features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                        # features = normalize_embeddings(features)
                        with features_history_lock:
                            features_his[id] = torch.cat([features_his[id], features.unsqueeze(0)], dim=0)
                            if features_his[id].size(0) > last_embedding_count:
                                features_his[id] = features_his[id][-last_embedding_count:]
                            # last_update_time = latest_current_time

                    # Draw bounding box and ID
                    cv2.circle(frame, (person_cx, person_cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
                    cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(frame, f"FPS-Count: {np.round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(exit_area, np.int32)], True, (0,0,255), 2)        
        cv2.imshow(f"YOLOv8 Tracking - {camera_path}", frame)

        # cv2.imwrite(f"YOLOv8 Tracking - {camera_path}.jpg", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


############################### Process Camera ###############################

def process_camera(camera_path, camera_id, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size):

    track_history = {}
    action_time = ''
    check_out_id_local = ""
    last_update_time = 0
    frame_count = 0
    model = YOLO(yolo_model_name)
    reid_model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    
    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # frame_count += 1
        # # Skip every other frame
        # if frame_count % 2 != 0:
        #     continue
        
        latest_current_time = time.time()
        frame = cv2.resize(frame, frame_size)
        results = model.track(frame, persist=True, stream=True)
        results = next(results)
        detection_status_check = len(results.boxes.data) > 0
        pose_boxes = results.boxes.data
        pose_keypoints = results.keypoints.xy
        
        if detection_status_check:
            for box, keypoint in zip(pose_boxes, pose_keypoints):
                x, y, w, h, pose_track_id, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(int(box[4])), box[5]

                person_cx = (x + w) // 2
                person_cy = (y + h) // 2

                right_hand = tuple(int(i) for i in keypoint[10])
                left_hand = tuple(int(i) for i in keypoint[9])

                
                features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                features = normalize_embeddings(features)

                with check_out_id_lock:
                    if check_out_id_local != check_out_id.value and check_out_id.value != "":
                        track_history = {k: v for k, v in track_history.items() if v != check_out_id.value}
                        check_out_id_local = check_out_id.value

                # If the person is new (not in track_history) and confidence is high

                if pose_track_id not in track_history and conf > 0.7:
                    # features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                    # features = normalize_embeddings(features)

                    best_match_id = None
                    best_similarity = 0

                    with features_history_lock:
                        for id, stored_embeddings in features_his.items():
                            # Compare the current features with stored embeddings
                            similarity_scores = torch.matmul(stored_embeddings, features)
                            # max_similarity = similarity_scores.max().item()
                            max_similarity = similarity_scores.mean()

                            # Keep track of the best match and ensure the similarity is above the threshold
                            if max_similarity > best_similarity and max_similarity >= similarity_threshold:
                                best_similarity = max_similarity
                                best_match_id = id

                        # If a match is found, update the track history and add new features
                        if best_match_id:
                            track_history[pose_track_id] = best_match_id
                            features_his[best_match_id] = torch.cat([features_his[best_match_id], features.unsqueeze(0)], dim=0)
                            # Keep only the last three embeddings
                            if features_his[best_match_id].size(0) > last_embedding_count:
                                features_his[best_match_id] = features_his[best_match_id][-last_embedding_count:]

                # Continues tracking
                if pose_track_id in track_history:
                    id = track_history[pose_track_id]
                    with weight_data_dict_lock:
                        if len(weight_data_dict) != 0:
                            for weight_action_time, weight_data in weight_data_dict.items():
                                if weight_action_time != action_time:
                                    if camera_id in weight_data['cameras']:
                                        key = f"{weight_action_time}_{camera_id}"
                                        value = {'frame': frame, 'session_id': id, 'pose_boxes':pose_boxes, 'pose_keypoints':pose_keypoints, 'camera_id': camera_id, 'location': weight_data['shelf_location'][camera_id], 'product': weight_data['product'], 'action_type':weight_data['action_type'], 'quantity':weight_data['quantity']}
                                        compress_value = compress_data(value)
                                        r_decision.set(key, compress_value)
                                    action_time = weight_action_time

                    if conf > 0.7:
                        # if latest_current_time - last_update_time >= 1:
                            # features = extract_features_from_frame_numpy(reid_model, frame, [x, y, w, h])
                            # features = normalize_embeddings(features)
                        with features_history_lock:
                            features_his[id] = torch.cat([features_his[id], features.unsqueeze(0)], dim=0)
                            if features_his[id].size(0) > last_embedding_count:
                                features_his[id] = features_his[id][-last_embedding_count:]
                            # last_update_time = latest_current_time

                    # Draw bounding box and ID
                    cv2.circle(frame, (person_cx, person_cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
                    cv2.circle(frame, right_hand, 4, (0, 0, 255), -1)
                    cv2.circle(frame, left_hand, 4, (255, 0, 0), -1)
                    cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(frame, f"FPS-Count: {np.round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"YOLOv8 Tracking - {camera_path}", frame)

        # cv2.imwrite(f"process_camera.jpg", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


############################### Fetch product Data ###############################

def fetching_product_data():
    while True:
        try:
            fetch_product_data(r_products)
            time.sleep(200000)
        except:
            time.sleep(10)
            continue


##################### Get data from weight sensor  and update dictionary #############################

def handle_weight_changes(client_socket, aisleId, sensorsPerLane, weight_data_dict, weight_data_dict_lock):
    # Initialize a dictionary to track state and history for each shelf
    shelves_state = create_shelves_stats()
    # print(shelves_state)
    
    MIN_WEIGHT_CHANGE = 5  # grams
    STABILIZATION_TIME = 1.0  # seconds
    MAX_WAIT_TIME = 3.0  # seconds
    WEIGHT_HISTORY_SIZE = 25
    STABILITY_THRESHOLD = 20000  # grams

    res_1 = setup_aisle_configuration(client_socket, aisleId, sensorsPerLane)
    print(f"{res_1} ------> aisle configuration")

    while True:
        try:
            # Get weights for all sensors
            sensor_weights = process_single_client(client_socket, aisleId)

            for rack_id in sensor_weights.keys():
                for shelf_id in sensor_weights[rack_id]['shelves'].keys():
                    shelf_data = sensor_weights[rack_id]['shelves'][shelf_id]

                    if isinstance(shelf_data, dict) and "sensors" in shelf_data:

                        # Calculate the total weight for the shelf by summing all sensor weights
                        total_shelf_weight = sum(shelf_data["sensors"])

                        # print(shelves_state)
                        shelf_state = shelves_state[f"{rack_id}_{shelf_id}"]
                        
                        # Add weight to history
                        shelf_state['weight_history'].append(total_shelf_weight)
                        if len(shelf_state['weight_history']) > WEIGHT_HISTORY_SIZE:
                            shelf_state['weight_history'].pop(0)

                        # If first measurement, initialize
                        if shelf_state["prev_weight"] is None:
                            shelf_state["prev_weight"] = total_shelf_weight
                            shelf_state["initial_weight"] = total_shelf_weight

                        weight_difference = total_shelf_weight - shelf_state["prev_weight"]

                        # Check if weight difference is significant
                        if abs(weight_difference) >= MIN_WEIGHT_CHANGE:
                            if shelf_state["state"] == "stable":
                                shelf_state["action_start_time"] = time.time()
                                shelf_state["state"] = "action"
                                shelf_state["initial_weight"] = shelf_state["prev_weight"]

                        current_time = time.time()
                        if shelf_state["state"] == "action":
                            if current_time - shelf_state["action_start_time"] >= STABILIZATION_TIME:

                                if abs(abs(max(shelf_state["weight_history"])) - abs(min(shelf_state["weight_history"]))) < STABILITY_THRESHOLD:
                                    total_change = total_shelf_weight - shelf_state["initial_weight"]

                                    action_type = "picked" if total_change < 0 else "dropped"
                                    product = identify_product(abs(total_change), shelf_data["products"])
                                    with weight_data_dict_lock:
                                        if product:
                                            print(f"Rack: {rack_id}, Shelf: {shelf_id}: Product: {product}, Action type: {action_type},  Shelf Location: {shelf_data['location']}, Cameras: {sensor_weights[rack_id]['cameras']}, action_time: {str(datetime.now().time())}, Weight change: {abs(total_change):.1f}g, Current weight: {total_shelf_weight:.1f}g")
                                            # weight_data_dict.clear()
                                            weight_data_dict.update({
                                                str(datetime.now().time()): {
                                                    'rack': rack_id,
                                                    'shelf': shelf_id,
                                                    'product': product,
                                                    'action_type': action_type,
                                                    'quantity': 1,
                                                    'shelf_location': shelf_data['location'],
                                                    'cameras': sensor_weights[rack_id]['cameras'],
                                                    'weight_change': abs(total_change),
                                                    "verify": True
                                                }
                                            })
                                        # if abs(total_change) > 10:
                                        #     weight_data_dict.update({
                                        #         str(datetime.now().time()): {
                                        #             'rack': rack_id,
                                        #             'shelf': shelf_id,
                                        #             'product': list(shelf_data["products"].keys()),
                                        #             'action_type': action_type,
                                        #             'quantity': 1,
                                        #             'shelf_location': shelf_data['location'],
                                        #             'cameras': sensor_weights[rack_id]['cameras'],
                                        #             'weight_change': abs(total_change),
                                        #             "verify": False
                                        #         }
                                        #     })
                                            

                                    shelf_state["state"] = "stable"
                                    shelf_state["weight_history"].clear()

                            elif current_time - shelf_state["action_start_time"] >= MAX_WAIT_TIME:
                                shelf_state["state"] = "stable"
                                shelf_state["weight_history"].clear()

                        # Update previous weight
                        shelf_state["prev_weight"] = total_shelf_weight

                        # print(f"Rack {rack_id}, Shelf {shelf_id}: Total weight on shelf: {total_shelf_weight}g")

        except Exception as e:
            print(f"Error: {e}")
            continue


################################# Decission Maker #################################################

def decission_maker(weight_data_dict, weight_data_dict_lock, product_original_name):
    obj_model = YOLO('yolov8n.pt').to('cuda')
    while True:
        try:
            decission_data_dictionary = decission_data(r_decision)
            if len(decission_data_dictionary) != 0:
                grouped_data = defaultdict(list)
                for key, value in decission_data_dictionary.items():
                    first_part = key.split('_')[0]
                    grouped_data[first_part].append({key: value})
                # Find all entries where the first part occurs more than once
                action_data = {k: v for k, v in grouped_data.items() if len(v) >= 1}
                if len(action_data) !=0:
                    products_near_hand = []
                    for data_key, data_value in action_data.items():
                        for data in data_value:
                            for inner_data_key, inner_data_values in data.items():
                                crop_image, crop_x, crop_y = capture_weight_area(inner_data_values['frame'], inner_data_values['location'])
                                for keypoint in inner_data_values['pose_keypoints']:
                                    right_hand = tuple(int(i) for i in keypoint[10])
                                    left_hand = tuple(int(i) for i in keypoint[9])
                                    right_elbow = tuple(int(i) for i in keypoint[8])   # Right elbow
                                    left_elbow = tuple(int(i) for i in keypoint[7])    # Left elbow
                                    palm_right_hand = extend_wrist_to_palm(right_elbow, right_hand, extension_distance=20)
                                    palm_left_hand = extend_wrist_to_palm(left_elbow, left_hand, extension_distance=20)

                                    hand_position = [palm_right_hand, palm_left_hand]

                                    for hand_index, position in enumerate(hand_position):
                                        relative_hand_position = (position[0] - crop_x, position[1] - crop_y)

                                        object_results = obj_model(crop_image, stream=True)
                                        object_results = next(object_results)

                                        object_boxes = object_results.boxes.data
                                        object_names = object_results.names

                                        if len(object_boxes) != 0:
                                            for obj_box in object_boxes:
                                                if object_names[int(obj_box[5])] in inner_data_values['product']:
                                                    obj_cls = object_names[int(obj_box[5])]
                                                    conf = obj_box[4]
                                                    obj_cx = int((obj_box[0] + obj_box[2]) // 2)
                                                    obj_cy = int((obj_box[1] + obj_box[3]) // 2)
                                                    dist_hand = calculate_distance(relative_hand_position, (obj_cx, obj_cy))
                                                    if dist_hand <= 60 and conf > 0.6:
                                                        products_near_hand.append(obj_cls)
                                                        print(f"product is near to hand and {inner_data_values['action_type']}")

                        # with decission_data_dictionary_lock:
                        delete_keys_with_prefix(r_decision, data_key) # remove from the data key from the weight data
                        with weight_data_dict_lock:
                            weight_data_dict.pop(data_key)

                    # # For combine resulte ----------------------
                    # if len(products_near_hand) != 0:
                    #     product = products_near_hand[0]
                    #     # product = product_original_name[product]
                    #     if inner_data_values['action_type'] == "picked":
                    #         if inner_data_values['session_id'] in picked_up_objects and product in picked_up_objects[inner_data_values['session_id']]:
                    #                 picked_up_objects[inner_data_values['session_id']][product] += 1
                    #                 picked_up_objects[inner_data_values['session_id']]['conf'] = 100
                    #         else:
                    #             if inner_data_values['session_id'] not in picked_up_objects:
                    #                 picked_up_objects[inner_data_values['session_id']] = {}
                    #             picked_up_objects[inner_data_values['session_id']][product] = 1
                    #             picked_up_objects[inner_data_values['session_id']]['conf'] = 100

                    #         product_data = load_product_data(r_products, product)
                    #         send_pickup_event('pickup', inner_data_values['session_id'], product_data)

                    #     else:
                    #         if inner_data_values['session_id'] in picked_up_objects and product in picked_up_objects[inner_data_values['session_id']]:
                    #             picked_up_objects[inner_data_values['session_id']][product] += -1
                    #             picked_up_objects[inner_data_values['session_id']]['conf'] = 100

                    #         product_data = load_product_data(r_products, product)
                    #         product_data = {"id":product_data['id']}
                    #         send_pickup_event('drop', inner_data_values['session_id'], product_data)

                    # For weight sensor based
                    # else:
                    product = inner_data_values['product'][0]
                    # product = product_original_name[product]
                    if inner_data_values['action_type'] == "picked":
                        if inner_data_values['session_id'] in picked_up_objects and product in picked_up_objects[inner_data_values['session_id']]:
                                picked_up_objects[inner_data_values['session_id']][product] += 1
                                picked_up_objects[inner_data_values['session_id']]['conf'] = 50
                        else:
                            if inner_data_values['session_id'] not in picked_up_objects:
                                picked_up_objects[inner_data_values['session_id']] = {}
                            picked_up_objects[inner_data_values['session_id']][product] = 1
                            picked_up_objects[inner_data_values['session_id']]['conf'] = 50

                        product_data = load_product_data(r_products, product)
                        send_pickup_event('pickup', inner_data_values['session_id'], product_data)
                        
                    else:
                        if inner_data_values['session_id'] in picked_up_objects and product in picked_up_objects[inner_data_values['session_id']]:
                            picked_up_objects[inner_data_values['session_id']][product] += -1
                            picked_up_objects[inner_data_values['session_id']]['conf'] = 50

                        product_data = load_product_data(r_products, product)
                        product_data = {"id":product_data['id']}
                        send_pickup_event('drop', inner_data_values['session_id'], product_data)

                    save_to_json(picked_up_objects, json_file)

        except Exception as e:
            print(e)
            continue



# "our own office"

# if __name__ == "__main__":

#     # Connect socket to send cart data
#     connect_to_server()

#     ############################### Global Variables ###############################
#     manager = multiprocessing.Manager()
#     features_his = manager.dict()
#     features_history_lock = multiprocessing.Lock()

#     weight_data_dict = manager.dict()
#     weight_data_dict_lock = multiprocessing.Lock()

#     global_track_count = manager.dict()
#     global_track_count_lock = multiprocessing.Lock()

#     check_out_id = manager.Value('c', "")
#     check_out_id_lock = multiprocessing.Lock()

#     # picked_up_objects = manager.dict()

#     product_original_name = {
#         "pringles": "Pringles Original 165 Gr.",
#         "wavylays": "Lay's Wavy BBQ Potato Chips",
#         "tapaltea": "Tapal Green Tea Lemon",
#         "surfexcel": "Surf Excel",
#         "spritehalf": "7up 500ml",
#         "spritecan": "Sprite Can (330ml)",
#         "olpers": "Olpers Milk",
#         "milkpack": "Nestle Milk Pack Nesvita",
#         "lifebouyshampo": "Lifebuoy Shampoo",
#         "hazeluntflavoredmilk": "Day Fresh Hazelnut Flavored Milk",
#         "fantacan": "Fanta Can 250ml",
#         "fantabottle": "Fanta Bottle 1.5ltr",
#         "cocacolabottle": "Coca Cola 1.5ltr",
#         "colgate": "Colgate ToothPaste",
#         "dettolsoap": "Dettol Soap (3-Pack)",
#         "pepsican": "Pepsi",
#         "britesurf": 'Brite Surf',
#         "closeup": 'CloseUp Toothpaste'
#         }

#     ############################### Entry & Exit poly area ###############################
#     exit_area = [(476, 317), (610, 391), (617, 538), (508, 462)]

#     # entry_area = [(738, 596), (717, 489), (132, 444), (228, 598)] # Entry for camera 5

#     entry_area = [(30, 15), (236, 0), (287, 281), (118, 350)]

#     # entry_area = [(46, 9), (211, -1), (251, 274), (129, 322)]

#     ############################### Yolo & reid model & Similarity Threshold & Last Embedding Count ###############################
#     yolo_model_name = "yolov8m-pose.pt"
#     model_path = 'repvgg_a0_person_reid_512.onnx'

#     similarity_threshold = 0.6
#     last_embedding_count = 3

#     ############################### Frame size and cmaera urls ###############################
#     frame_size = (1080, 600)

#     # camera urls
#     # entry_cam = "rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=5&subtype=0"
#     entry_cam = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=1&subtype=0"
#     exit_cam = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=4&subtype=0"
#     # exit_cam = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=1&subtype=0"
#     process_cam7 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=7&subtype=0"
#     process_cam3 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=3&subtype=0"
#     process_cam6 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=6&subtype=0"

#     # entry_cam = r"C:\Users\mmoht\Desktop\camera_calibration\cam1.mp4"
#     # process_cam7 = r"C:\Users\mmoht\Desktop\camera_calibration\cam4.mp4"
#     # process_cam3 = r"C:\Users\mmoht\Desktop\camera_calibration\cam6.mp4"

#     ############################### Weight sensor variables and weight socket connection ###############################
#     # GET_WEIGHT = [0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03]
#     # TARE = [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]



#     # Old code directly connected with weight sensor
#     # aisle_Id = 4
#     # sensors_Per_Lane = 2
#     # HOST = '192.168.1.220'
#     # PORT = 8235

#     # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     # server_socket.bind((HOST, PORT))
#     # server_socket.listen(2)
#     # client_socket, addr = server_socket.accept()
#     # print("Socket connected", client_socket)
    

#     # connect with raspi 
#     # aisle_Id = 4
#     # sensors_Per_Lane = 2

   

#    # connect with raspi 
#     aisle_Id = 3
#     sensors_Per_Lane = 2

#     HOST = '59.103.120.15'
#     PORT = 8077
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     # Connect to the server
#     client_socket.connect((HOST, PORT))
#     print(f"Connected to server at {HOST}:{PORT}")

#     ############################ Define Processes ###############################

#     # multiprocessing.set_start_method('spawn', force=True)

#     entry_process = multiprocessing.Process(target=entry_camera, args=(entry_cam, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, entry_area, frame_size,))
    
#     exit_process = multiprocessing.Process(target=exit_camera, args=(exit_cam, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, exit_area, frame_size,))

#     process_process = multiprocessing.Process(target=process_camera, args=(process_cam7, 7, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))
    
#     process_process_2 = multiprocessing.Process(target=process_camera, args=(process_cam3, 3, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))

#     # process_process_3 = multiprocessing.Process(target=process_camera, args=(process_cam6, 6, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))


#     fetching_product_data_process = multiprocessing.Process(target=fetching_product_data)
    
#     weight_process = multiprocessing.Process(target=handle_weight_changes, args=(client_socket, aisle_Id, sensors_Per_Lane, weight_data_dict, weight_data_dict_lock,))

#     decission_process = multiprocessing.Process(target=decission_maker, args=(weight_data_dict, weight_data_dict_lock, product_original_name,))

#     ############################ start Processes ###############################

#     entry_process.start()

#     exit_process.start()

#     process_process.start()

#     process_process_2.start()

#     # process_process_3.start()

#     fetching_product_data_process.start()

#     weight_process.start()

#     decission_process.start()

#     ############################ join Processes ###############################
#     entry_process.join()

#     exit_process.join()

#     process_process.join()

#     process_process_2.join()

#     # process_process_3.join()

#     fetching_product_data_process.join()

#     weight_process.join()

#     decission_process.join()



# "Harish sahb office"

# if __name__ == "__main__":

#     # Connect socket to send cart data
#     connect_to_server()

#     ############################### Global Variables ###############################
#     manager = multiprocessing.Manager()
#     features_his = manager.dict()
#     features_history_lock = multiprocessing.Lock()

#     weight_data_dict = manager.dict()
#     weight_data_dict_lock = multiprocessing.Lock()

#     global_track_count = manager.dict()
#     global_track_count_lock = multiprocessing.Lock()

#     check_out_id = manager.Value('c', "")
#     check_out_id_lock = multiprocessing.Lock()

#     # picked_up_objects = manager.dict()

#     product_original_name = {
#         "pringles": "Pringles Original 165 Gr.",
#         "wavylays": "Lay's Wavy BBQ Potato Chips",
#         "tapaltea": "Tapal Green Tea Lemon",
#         "surfexcel": "Surf Excel",
#         "spritehalf": "7up 500ml",
#         "spritecan": "Sprite Can (330ml)",
#         "olpers": "Olpers Milk",
#         "milkpack": "Nestle Milk Pack Nesvita",
#         "lifebouyshampo": "Lifebuoy Shampoo",
#         "hazeluntflavoredmilk": "Day Fresh Hazelnut Flavored Milk",
#         "fantacan": "Fanta Can 250ml",
#         "fantabottle": "Fanta Bottle 1.5ltr",
#         "cocacolabottle": "Coca Cola 1.5ltr",
#         "colgate": "Colgate ToothPaste",
#         "dettolsoap": "Dettol Soap (3-Pack)",
#         "pepsican": "Pepsi",
#         "britesurf": 'Brite Surf',
#         "closeup": 'CloseUp Toothpaste'
#         }


#     ############################### Entry & Exit poly area ###############################
   
#     entry_area = [(506, 215), (682, 213), (685, 340), (509, 338)]
#     # exit_area = [(519, 63), (583, 543), (237, 528), (248, 192)]
#     exit_area = [(372, 255), (577, 247), (583, 511), (400, 507)]

#     ############################### Yolo & reid model & Similarity Threshold & Last Embedding Count ###############################
#     yolo_model_name = "yolov8m-pose.pt"
#     model_path = 'repvgg_a0_person_reid_512.onnx'

#     similarity_threshold = 0.65
#     last_embedding_count = 3

#     ############################### Frame size and cmaera urls ###############################
#     frame_size = (1080, 600)


#     entry_cam = 'rtsp://admin:Haris0202@80.251.198.12:554/Streaming/Channels/301'
#     exit_cam = 'rtsp://admin:Haris0202@80.251.198.12:554/Streaming/Channels/201'
#     process_cam1 = 'rtsp://admin:Haris0202@80.251.198.12:554/Streaming/Channels/401'
#     process_cam2 = 'rtsp://admin:Haris0202@80.251.198.12:554/Streaming/Channels/501'

#     ############################### Weight sensor variables and weight socket connection ###############################
#     # GET_WEIGHT = [0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03]
#     # TARE = [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]



#     # connect with raspi 
#     aisle_Id = 3
#     sensors_Per_Lane = 2


#     HOST = '80.251.198.12'
#     PORT = 8080
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     # Connect to the server
#     client_socket.connect((HOST, PORT))
#     print(f"Connected to server at {HOST}:{PORT}")

#     ############################ Define Processes ###############################

#     # multiprocessing.set_start_method('spawn', force=True)

#     entry_process = multiprocessing.Process(target=entry_camera, args=(entry_cam, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, entry_area, frame_size,))
    
#     exit_process = multiprocessing.Process(target=exit_camera, args=(exit_cam, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, exit_area, frame_size,))

#     # process_process = multiprocessing.Process(target=process_camera, args=(process_cam1, 401, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))
    
#     process_process_2 = multiprocessing.Process(target=process_camera, args=(process_cam2, 501, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))

#     # process_process_3 = multiprocessing.Process(target=process_camera, args=(process_cam6, 6, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))


#     fetching_product_data_process = multiprocessing.Process(target=fetching_product_data)
    
#     weight_process = multiprocessing.Process(target=handle_weight_changes, args=(client_socket, aisle_Id, sensors_Per_Lane, weight_data_dict, weight_data_dict_lock,))

#     decission_process = multiprocessing.Process(target=decission_maker, args=(weight_data_dict, weight_data_dict_lock, product_original_name,))

#     ############################ start Processes ###############################

#     entry_process.start()

#     exit_process.start()

#     # process_process.start()

#     process_process_2.start()

#     # process_process_3.start()

#     fetching_product_data_process.start()

#     weight_process.start()

#     decission_process.start()

#     ############################ join Processes ###############################
#     entry_process.join()

#     exit_process.join()

#     # process_process.join()

#     process_process_2.join()

#     # process_process_3.join()

#     fetching_product_data_process.join()

#     weight_process.join()

#     decission_process.join()


# Vester Hasienge shop

if __name__ == "__main__":

    # Connect socket to send cart data
    connect_to_server()

    ############################### Global Variables ###############################
    manager = multiprocessing.Manager()
    features_his = manager.dict()
    features_history_lock = multiprocessing.Lock()

    weight_data_dict = manager.dict()
    weight_data_dict_lock = multiprocessing.Lock()

    global_track_count = manager.dict()
    global_track_count_lock = multiprocessing.Lock()

    check_out_id = manager.Value('c', "")
    check_out_id_lock = multiprocessing.Lock()

    # picked_up_objects = manager.dict()

    product_original_name = {
        "pringles": "Pringles Original 165 Gr.",
        "wavylays": "Lay's Wavy BBQ Potato Chips",
        "tapaltea": "Tapal Green Tea Lemon",
        "surfexcel": "Surf Excel",
        "spritehalf": "7up 500ml",
        "spritecan": "Sprite Can (330ml)",
        "olpers": "Olpers Milk",
        "milkpack": "Nestle Milk Pack Nesvita",
        "lifebouyshampo": "Lifebuoy Shampoo",
        "hazeluntflavoredmilk": "Day Fresh Hazelnut Flavored Milk",
        "fantacan": "Fanta Can 250ml",
        "fantabottle": "Fanta Bottle 1.5ltr",
        "cocacolabottle": "Coca Cola 1.5ltr",
        "colgate": "Colgate ToothPaste",
        "dettolsoap": "Dettol Soap (3-Pack)",
        "pepsican": "Pepsi",
        "britesurf": 'Brite Surf',
        "closeup": 'CloseUp Toothpaste'
        }


    ############################### Entry & Exit poly area ###############################
   
    entry_area = [(369, 379), (693, 395), (689, 442), (358, 417)]
    # exit_area = [(519, 63), (583, 543), (237, 528), (248, 192)]
    exit_area = [(608, 600), (432, 560), (673, 372), (742, 417)]

    ############################### Yolo & reid model & Similarity Threshold & Last Embedding Count ###############################
    yolo_model_name = "yolov8m-pose.pt"
    model_path = 'repvgg_a0_person_reid_512.onnx'

    similarity_threshold = 0.65
    last_embedding_count = 3

    ############################### Frame size and cmaera urls ###############################
    frame_size = (1080, 600)


    entry_cam = 'rtsp://admin:Admin12345*@5.103.180.194:554/cam/realmonitor?channel=8&subtype=0'
    exit_cam = 'rtsp://admin:Admin12345*@5.103.180.194:554/cam/realmonitor?channel=18&subtype=0'
    process_cam1 = 'rtsp://admin:Admin12345*@5.103.180.194:554/cam/realmonitor?channel=21&subtype=0'
    process_cam2 = 'rtsp://admin:Admin12345*@5.103.180.194:554/cam/realmonitor?channel=9&subtype=0'

    ############################### Weight sensor variables and weight socket connection ###############################
    # GET_WEIGHT = [0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03]
    # TARE = [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]



    # connect with raspi 
    aisle_Id = 9
    sensors_Per_Lane = 2


    HOST = '5.103.180.194'
    PORT = 8087
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

    ############################ Define Processes ###############################

    # multiprocessing.set_start_method('spawn', force=True)

    entry_process = multiprocessing.Process(target=entry_camera, args=(entry_cam, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, entry_area, frame_size,))
    
    exit_process = multiprocessing.Process(target=exit_camera, args=(exit_cam, yolo_model_name, similarity_threshold, features_his, features_history_lock, global_track_count, global_track_count_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, exit_area, frame_size,))

    # process_process = multiprocessing.Process(target=process_camera, args=(process_cam1, 2, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))
    
    process_process_2 = multiprocessing.Process(target=process_camera, args=(process_cam2, 9, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))

    # process_process_3 = multiprocessing.Process(target=process_camera, args=(process_cam6, 21, yolo_model_name, similarity_threshold, features_his, features_history_lock, weight_data_dict, weight_data_dict_lock, check_out_id, check_out_id_lock, last_embedding_count, model_path, frame_size,))


    fetching_product_data_process = multiprocessing.Process(target=fetching_product_data)
    
    weight_process = multiprocessing.Process(target=handle_weight_changes, args=(client_socket, aisle_Id, sensors_Per_Lane, weight_data_dict, weight_data_dict_lock,))

    decission_process = multiprocessing.Process(target=decission_maker, args=(weight_data_dict, weight_data_dict_lock, product_original_name,))

    ############################ start Processes ###############################

    entry_process.start()

    exit_process.start()

    # process_process.start()

    process_process_2.start()

    # process_process_3.start()

    fetching_product_data_process.start()

    weight_process.start()

    decission_process.start()

    ############################ join Processes ###############################
    entry_process.join()

    exit_process.join()

    # process_process.join()

    process_process_2.join()

    # process_process_3.join()

    fetching_product_data_process.join()

    weight_process.join()

    decission_process.join()



