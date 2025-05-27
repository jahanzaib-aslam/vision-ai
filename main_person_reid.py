# import multiprocessing
import torchreid
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import torch.nn.functional as F
from ultralytics import YOLO
import sys
import os
import time
import torch.multiprocessing as mp
import redis
import random
import numpy as np
import json
from datetime import datetime
from weight_sensor import *
from helping_function import *



mp.set_start_method('spawn', force=True)

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reid_data = redis.Redis(host='127.0.0.1', port=6379, db=0)
decission_weight_data = redis.Redis(host='127.0.0.1', port=6379, db=1)
print("Flush old data", reid_data.flushall())
print("decission old data", decission_weight_data.flushall())


def tensor_to_bytes(tensor):
    return tensor.cpu().numpy().tobytes()

def bytes_to_tensor(byte_data, shape):
    array = np.frombuffer(byte_data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(array)

def get_all_embeddings_from_redis(redis_conn, key, shape):
    byte_list = redis_conn.lrange(key, 0, -1)
    tensors = [bytes_to_tensor(b, shape) for b in byte_list]
    return torch.stack(tensors).to(device)

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

def generate_id():
    return random.randint(1000, 9999)

def load_transform():
    return T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.4364566, 0.40887514, 0.4093984],
                    std=[0.25174066, 0.24528353, 0.23656533])
    ])

def load_reid_model():
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=0,
        pretrained=False
    )
    model = model.to(device)
    model.eval()
    torchreid.utils.load_pretrained_weights(model, "/home/muhtashim/Desktop/reid-test/finetune_osnetV1/model.pth.tar-15")
    return model

def entry_camera(camera_path, similarity_threshold, last_embedding_count, entry_area, frame_size):
    
    track_history = {}
    last_coming_id = None

    transform = load_transform()
    model = load_reid_model()
    yolo_model = YOLO("/home/muhtashim/Desktop/reid-test/yolov8n-pose.engine")
    # yolo_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.resize(frame, frame_size)

        results = yolo_model.track(frame, persist=True, stream=True)
        results = next(results)
        detection_status_check = len(results.boxes.data) > 0
        pose_boxes = results.boxes.data
        pose_keypoints = results.keypoints.xy

        if detection_status_check:
            batch_crops = []
            for box in pose_boxes:
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                person_crop = frame[y:h, x:w]
                img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                img = transform(img)
                batch_crops.append(img)

            batch_tensor = torch.stack(batch_crops).to(device)
            with torch.no_grad():
                features = model(batch_tensor)
            features = F.normalize(features, dim=1)

            for box, keypoint, feature in zip(pose_boxes, pose_keypoints, features):
                x, y, w, h, pose_track_id, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(int(box[4])), box[5]

                person_cx = (x + w) // 2
                person_cy = (y + h) // 2
            

                # Check if the person is in the entry area
                if entry_area and cv2.pointPolygonTest(np.array(entry_area, np.int32), (person_cx, person_cy), False) >= 0 and conf > 0.7:
                    if pose_track_id not in track_history:
                        random_id = generate_id()

                        base_id = random_id
                        if last_coming_id is None or not last_coming_id.startswith(base_id):
                            random_id = f"{base_id}_1"
                        else:
                            suffix = int(last_coming_id.split('_')[1]) + 1
                            random_id = f"{base_id}_{suffix}"
                        last_coming_id = random_id

                        track_history[pose_track_id] = random_id
                        reid_data.rpush(random_id, tensor_to_bytes(feature))
                      

                # If the person is new (not in track_history) and confidence is high
                if pose_track_id not in track_history and conf > 0.7:
                    best_match_id = None
                    best_similarity = 0

                    for k in reid_data.keys("*"):
                        id = k.decode()
                        stored_embeddings = get_all_embeddings_from_redis(reid_data, id, feature.shape)

                        similarity_scores = torch.matmul(stored_embeddings, feature)
                        max_similarity = similarity_scores.mean()

                        if max_similarity > best_similarity and max_similarity >= similarity_threshold:
                            best_similarity = max_similarity
                            best_match_id = id
                                    
                    if best_match_id:
                        track_history[pose_track_id] = best_match_id
                        reid_data.rpush(best_match_id, tensor_to_bytes(feature))
                        reid_data.ltrim(best_match_id, -last_embedding_count, -1)


                if pose_track_id in track_history:
                    id = track_history[pose_track_id]
                    if conf > 0.7:
                        reid_data.rpush(id, tensor_to_bytes(feature))
                        reid_data.ltrim(id, -last_embedding_count, -1)

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

def process_camera(camera_path, camera_id, similarity_threshold, last_embedding_count, frame_size):
    
    track_history = {}
    capture_state = {}
    frame_counter = {}
    capture_decission_frame_data = {}

    transform = load_transform()
    model = load_reid_model()
    yolo_model = YOLO("/home/muhtashim/Desktop/reid-test/yolov8n-pose.engine")
    # yolo_model = YOLO("yolov8n-pose.pt")


    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.resize(frame, frame_size)
        predict_results = yolo_model.track(frame, persist=True, stream=True)
        results = next(predict_results)
        detection_status_check = len(results.boxes.data) > 0
        pose_boxes = results.boxes.data
        pose_keypoints = results.keypoints.xy

        if detection_status_check:
            batch_crops = []
            for box in pose_boxes:
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                person_crop = frame[y:h, x:w]
                img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                img = transform(img)
                batch_crops.append(img)

            batch_tensor = torch.stack(batch_crops).to(device)
            with torch.no_grad():
                features = model(batch_tensor)
            features = F.normalize(features, dim=1)

            all_ids = {"ids": {}, "frame": frame}
            for box, keypoint, feature in zip(pose_boxes, pose_keypoints, features):
                x, y, w, h, pose_track_id, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(int(box[4])), box[5]

                points = [tuple(map(int, keypoint[i])) for i in [10, 9, 8, 7]]
                right_hand, left_hand, right_elbow, left_elbow = points

                palm_right_hand = extend_wrist_to_palm(right_elbow, right_hand, extension_distance=20)
                palm_left_hand = extend_wrist_to_palm(left_elbow, left_hand, extension_distance=20)

            
                # If the person is new (not in track_history) and confidence is high
                if pose_track_id not in track_history and conf > 0.7:
                    best_match_id = None
                    best_similarity = 0

                    for k in reid_data.keys("*"):
                        id = k.decode()
                        stored_embeddings = get_all_embeddings_from_redis(reid_data, id, feature.shape)

                        similarity_scores = torch.matmul(stored_embeddings, feature)
                        max_similarity = similarity_scores.mean()

                        if max_similarity > best_similarity and max_similarity >= similarity_threshold:
                            best_similarity = max_similarity
                            best_match_id = id
                                    
                    if best_match_id:
                        track_history[pose_track_id] = best_match_id
                        reid_data.rpush(best_match_id, tensor_to_bytes(feature))
                        reid_data.ltrim(best_match_id, -last_embedding_count, -1)

                if pose_track_id in track_history:
                    id = track_history[pose_track_id]
                    all_ids["ids"][id] = [palm_right_hand, palm_left_hand]

                    if conf > 0.7:
                        reid_data.rpush(id, tensor_to_bytes(feature))
                        reid_data.ltrim(id, -last_embedding_count, -1)

                    # Draw bounding box and ID
                    # cv2.circle(frame, (person_cx, person_cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
                    cv2.circle(frame, palm_right_hand, 4, (0, 0, 255), -1)
                    cv2.circle(frame, palm_left_hand, 4, (255, 0, 0), -1)
                    cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # weight trigger
            weight_data_dict = decission_weight_data.hgetall("weight_data")
            if len(weight_data_dict) != 0:
                for key, value in weight_data_dict.items():
                    weight_action_time = key.decode()
                    weight_data_entry = json.loads(value.decode())

                    if weight_action_time not in capture_state:

                        capture_state[weight_action_time] = True
                        frame_counter[weight_action_time] = 0
                        capture_decission_frame_data[weight_action_time] = {}

                    if capture_state[weight_action_time]:
                        if frame_counter[weight_action_time] < 10:
                            capture_decission_frame_data[weight_action_time][f"frame_{frame_counter[weight_action_time]}"] = all_ids
                            frame_counter[weight_action_time] += 1

                        else:
                            capture_state[weight_action_time] = False
                            value = {"frames_info": capture_decission_frame_data, 'camera_id': camera_id, 'location': weight_data_entry['shelf_location'][camera_id], 'product': weight_data_entry['product'], 'action_type':weight_data_entry['action_type'], 'quantity':weight_data_entry['quantity'], "weight_change": weight_data_entry['weight_change'], "verify": weight_data_entry['verify']}
                            compress_value = compress_data(value)
                            decission_weight_data.hset("decission_data", f"{weight_action_time}_{camera_id}", compress_value)
                             

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(frame, f"FPS-Count: {np.round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"YOLOv8 Tracking - {camera_path}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def handle_weight_changes(client_socket, aisleId, sensorsPerLane):
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

                                    if product:
                                        print(f"Rack: {rack_id}, Shelf: {shelf_id}: Product: {product}, Action type: {action_type},  Shelf Location: {shelf_data['location']}, Cameras: {sensor_weights[rack_id]['cameras']}, action_time: {str(datetime.now().time())}, Weight change: {abs(total_change):.1f}g, Current weight: {total_shelf_weight:.1f}g")
                                        timestamp_key = str(datetime.now().time())
                                        entry = {
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
                                        decission_weight_data.hset("weight_data", timestamp_key, json.dumps(entry))

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


if __name__ == "__main__":

    entry_area = [(30, 15), (236, 0), (287, 281), (118, 350)]

    similarity_threshold = 0.7
    last_embedding_count = 50

    ############################### Frame size and cmaera urls ###############################
    frame_size = (1080, 600)

    # camera urls
    # entry_cam = "rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=5&subtype=0"
    entry_cam = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=1&subtype=0"
    exit_cam = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=4&subtype=0"
    # exit_cam = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=1&subtype=0"
    process_cam7 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=7&subtype=0"
    process_cam3 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=3&subtype=0"

    process_cam5 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=5&subtype=0"
    process_cam6 = "rtsp://dev:Admin@1234@59.103.120.15:554/cam/realmonitor?channel=6&subtype=0"



    entry_process = mp.Process(target=entry_camera, args=(entry_cam, similarity_threshold, last_embedding_count, entry_area, frame_size,))
    
    process_process = mp.Process(target=process_camera, args=(process_cam5, 5, similarity_threshold, last_embedding_count, frame_size,))
    
    process_process_2 = mp.Process(target=process_camera, args=(process_cam3, 3, similarity_threshold, last_embedding_count, frame_size,))

    process_process_3 = mp.Process(target=process_camera, args=(exit_cam, 4, similarity_threshold, last_embedding_count, frame_size,))


    ############################ start Processes ###############################

    entry_process.start()

    process_process.start()

    process_process_2.start()

    process_process_3.start()

    ############################ join Processes ###############################
    entry_process.join()

    process_process.join()

    process_process_2.join()

    process_process_3.join()







