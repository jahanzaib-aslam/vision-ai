import cv2
import numpy as np
import json
import os
import time
import torch
from ultralytics import YOLO
import cvzone
import random
import requests
import threading
from backed_api import fetch_product_data, get_userId, check_out, load_product_data
from object_detection import process_frame
from socket_connection import connect_to_server, send_pickup_event
from datetime import datetime
from helping_function import *
from weight_sensor import *
import socket
from collections import defaultdict
import redis

r_decision = redis.Redis(host='127.0.0.1', port=6379, db=0)
r_products = redis.Redis(host='127.0.0.1', port=6379, db=1)

connect_to_server()
torch.cuda.set_device(0)

############################### Initial defines ###############################

picked_up_objects = {}

json_file = "picked_up_objects.json"
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        picked_up_objects = json.load(f)

############################### Global Variables ###############################


weight_data_dict_lock = threading.Lock()
weight_data_dict = {}


main_session_id = 0
person_count = 0

detection_status = {"rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=1&subtype=0":True}
detection_lock = threading.Lock()

############################### Entry camera function ###############################


def entry_camera(camera_path, yolo_model_name, entry_area, frame_size):
    global person_count
    global main_session_id
    global detection_status

    track_history= {}
    model = YOLO(yolo_model_name)

    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.resize(frame, frame_size)
        if not success:
            break
        
        detection_status_check, pose_boxes, pose_keypoints = process_frame(frame, model)

        if main_session_id != 0:
            with detection_lock:
                    detection_status[camera_path] = detection_status_check

        if detection_status_check == True:

            for box, keypoint in zip(pose_boxes, pose_keypoints):
                # x, y, w, h, track_id, conf = box[0], box[1], box[2], box[3], str(int(box[4])), box[5]
                x, y, w, h, conf = box[0], box[1], box[2], box[3], box[4]

                if conf > 0.5:
                    person_cx=int(int(x)+int(w))//2
                    person_cy=int(int(y)+int(h))//2

                    right_hand = tuple(int(i) for i in keypoint[10])
                    left_hand = tuple(int(i) for i in keypoint[9])

                    if main_session_id == 0:
                        # if cv2.pointPolygonTest(np.array(entry_area, np.int32), (person_cx, person_cy), False) >= 0 and track_id not in list(track_history.keys()):
                        if cv2.pointPolygonTest(np.array(entry_area, np.int32), (person_cx, person_cy), False) >= 0:
                            # session_id = get_userId()
                            session_id = generate_random_id()
                            # print("id", session_id)
                            # track_history[track_id] = session_id
                            main_session_id = session_id
                            # with  person_count_lock:
                            #     person_count+=1

                        # Draw bounding box and ID
                    cv2.circle(frame, (person_cx, person_cy), 4, (0, 0, 255), -1)
                    cv2.circle(frame, right_hand, 4, (0, 0, 255), -1)
                    cv2.circle(frame, left_hand, 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), 2)
                    cv2.putText(frame, str(main_session_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        cv2.putText(frame, f"FPS-Count: {np.round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"person count {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(entry_area, np.int32)], True, (255,0,0), 2)
        # cv2.polylines(frame, [np.array(shelf_one_area, np.int32)], True, (255, 0, 0), 2)      
        cv2.imshow(f"YOLOv8 Tracking - {camera_path}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



############################### Process Camera ###############################


def process_camera(camera_path, camera_id, yolo_model_name, frame_size):
    global person_count
    global main_session_id
    global detection_status
    global weight_data_dict
    action_time = ''
    model = YOLO(yolo_model_name)

    cap = cv2.VideoCapture(camera_path)
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.resize(frame, frame_size)
        if not success:
            break

        detection_status_check, pose_boxes, pose_keypoints = process_frame(frame, model)

        if main_session_id != 0:
            with detection_lock:
                    detection_status[camera_path] = detection_status_check

        if detection_status_check == True:
            
            with weight_data_dict_lock:
                if action_time not in weight_data_dict and len(weight_data_dict) != 0:
                    for weight_action_time, weight_data in weight_data_dict.items():
                        if camera_id in weight_data['cameras']:
                            key = f"{weight_action_time}_{camera_id}"
                            value = {'frame': frame, 'session_id': main_session_id, 'pose_boxes':pose_boxes, 'pose_keypoints':pose_keypoints, 'camera_id': camera_id, 'location': weight_data['shelf_location'][camera_id], 'product': weight_data['product'], 'action_type':weight_data['action_type'], 'quantity':weight_data['quantity']}
                            compress_value = compress_data(value)
                            r_decision.set(key, compress_value)
                        action_time = weight_action_time

            for box, keypoint in zip(pose_boxes, pose_keypoints):
                # x, y, w, h, track_id = box[0], box[1], box[2], box[3], str(int(box[4]))
                x, y, w, h, conf = box[0], box[1], box[2], box[3], box[4]

                person_cx=int(int(x)+int(w))//2
                person_cy=int(int(y)+int(h))//2

                right_hand = tuple(int(i) for i in keypoint[10])
                left_hand = tuple(int(i) for i in keypoint[9])
                # hand_position = [right_hand, left_hand]

                # Draw bounding box and ID
                cv2.circle(frame, (person_cx, person_cy), 4, (0, 0, 255), -1)
                cv2.circle(frame, right_hand, 4, (0, 0, 255), -1)
                cv2.circle(frame, left_hand, 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), 2)
                cv2.putText(frame, str(main_session_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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



############################### checkout scenario ###############################

def checkout_scenario():
    global main_session_id
    global detection_status
    try:
        time.sleep(10)
        while True:
            with detection_lock:
                if not any(detection_status.values()):
                    if main_session_id not in picked_up_objects:
                        picked_up_objects[main_session_id] = {}
                    
                    # check_out(main_session_id)
                    picked_up_objects[main_session_id]["Person checkout"] = "True"
                    save_to_json(picked_up_objects, json_file)
                    main_session_id = 0
                    detection_status["rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=1&subtype=0"]=True
            time.sleep(1)  # Adjust sleep time as needed
    except KeyboardInterrupt:
        print("Stopping...")


############################### Fetch product Data ###############################

def fetching_product_data():
    while True:
        try:
            fetch_product_data(r_products)
            time.sleep(20)
        except:
            time.sleep(10)
            continue


##################### Get data from weight sensor  and update dictionary #############################

def handle_weight_changes(client_socket, GET_WEIGHT, TARE):
    # Initialize a dictionary to track state and history for each shelf
    global weight_data_dict
    shelves_state = create_shelves_stats()
    
    MIN_WEIGHT_CHANGE = 5  # grams
    STABILIZATION_TIME = 1.0  # seconds
    MAX_WAIT_TIME = 3.0  # seconds
    WEIGHT_HISTORY_SIZE = 25
    STABILITY_THRESHOLD = 2  # grams

    # Perform TARE operation for each sensor
    send_command_and_handle_response(client_socket, TARE)

    while True:
        try:
            # Get weights for all sensors
            sensor_weights = send_command_and_handle_response(client_socket, GET_WEIGHT)
            
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
                                # Check if weight has stabilized
                                if abs(max(shelf_state["weight_history"])) - abs(min(shelf_state["weight_history"])) < STABILITY_THRESHOLD:
                                    total_change = total_shelf_weight - shelf_state["initial_weight"]
                                    action_type = "picked" if total_change < 0 else "dropped"
                                    product = identify_product(abs(total_change), shelf_data["products"])
                                    if product:
                                        # print(f"Rack: {rack_id}, Shelf: {shelf_id}: Product: {product}, Action type: {action_type},  Shelf Location: {shelf_data['location']}, Cameras: {sensor_weights[rack_id]['cameras']}, action_time: {str(datetime.now().time())}, Weight change: {abs(total_change):.1f}g, Current weight: {total_shelf_weight:.1f}g")
                                        with weight_data_dict_lock:
                                            weight_data_dict = {
                                                str(datetime.now().time()): {
                                                    'rack': rack_id,
                                                    'shelf': shelf_id,
                                                    'product': product,
                                                    'action_type': action_type,
                                                    'quantity': 1,
                                                    'shelf_location': shelf_data['location'],
                                                    'cameras': sensor_weights[rack_id]['cameras'],
                                                }
                                            }
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

def decission_maker():
    obj_model = YOLO('yolov8x.pt').to('cuda')
    while True:
        try:
            decission_data_dictionary = decission_data(r_decision)
            if len(decission_data_dictionary) != 0:
                grouped_data = defaultdict(list)
                for key, value in decission_data_dictionary.items():
                    first_part = key.split('_')[0]
                    grouped_data[first_part].append({key: value})
                # Find all entries where the first part occurs more than once
                action_data = {k: v for k, v in grouped_data.items() if len(v) > 1}
                if len(action_data) !=0:
                    products_near_hand = []
                    for data_key, data_value in action_data.items():
                        for data in data_value:
                            for inner_data_key, inner_data_values in data.items():
                                crop_image, crop_x, crop_y = capture_weight_area(inner_data_values['frame'], inner_data_values['location'])
                                for keypoint in inner_data_values['pose_keypoints']:
                                    right_hand = tuple(int(i) for i in keypoint[10])
                                    left_hand = tuple(int(i) for i in keypoint[9])
                                    hand_position = [right_hand, left_hand]

                                    # cv2.circle(crop_image, (inner_data_values['location'][0] - crop_x, inner_data_values['location'][1] - crop_y), 4, (155, 0, 255), -1)
                                    # cv2.circle(crop_image, (hand_position[0][0] - crop_x, hand_position[0][1] - crop_y), 4, (0, 255, 255), -1)
                                    # cv2.circle(crop_image, (hand_position[1][0] - crop_x, hand_position[1][1] - crop_y), 4, (0, 255, 0), -1)
                                    
                                    for hand_index, position in enumerate(hand_position):
                                        relative_hand_position = (position[0] - crop_x, position[1] - crop_y)

                                        object_results = obj_model(crop_image, stream=True)
                                        object_results = next(object_results)

                                        # if hand_index == 0:
                                        #     cv2.imshow(f"{inner_data_key}_Right Hand output", object_results.plot())
                                        # else:
                                        #     cv2.imshow(f"{inner_data_key}_Left Hand output", object_results.plot())
                                    
                                        object_boxes = object_results.boxes.data
                                        object_names = object_results.names

                                        if len(object_boxes) != 0:
                                            for obj_box in object_boxes:
                                                if object_names[int(obj_box[5])] in " ".join([inner_data_values['product']]):
                                                    obj_cls = object_names[int(obj_box[5])]
                                                    conf = obj_box[4]
                                                    obj_cx = int((obj_box[0] + obj_box[2]) // 2)
                                                    obj_cy = int((obj_box[1] + obj_box[3]) // 2)
                                                    dist_hand = calculate_distance(relative_hand_position, (obj_cx, obj_cy))
                                                    if dist_hand <= 50 and conf > 0.5:
                                                        products_near_hand.append(obj_cls)
                                                        print(f"product is near to hand and {inner_data_values['action_type']}")

                        # with decission_data_dictionary_lock:
                        delete_keys_with_prefix(r_decision, data_key) # remove from the data key from the weight data
                    
                    # For combine resulte ----------------------
                    if len(products_near_hand) != 0:
                        product = products_near_hand[0]
                        if inner_data_values['action_type'] == "picked":
                            if inner_data_values['session_id'] in picked_up_objects and product in picked_up_objects[inner_data_values['session_id']]:
                                    picked_up_objects[inner_data_values['session_id']][product] += 1
                                    picked_up_objects[inner_data_values['session_id']]['conf'] = 100
                            else:
                                if inner_data_values['session_id'] not in picked_up_objects:
                                    picked_up_objects[inner_data_values['session_id']] = {}
                                picked_up_objects[inner_data_values['session_id']][product] = 1
                                picked_up_objects[inner_data_values['session_id']]['conf'] = 100

                            # product_data = load_product_data(r_products, product)
                            # send_pickup_event('pickup', inner_data_values['session_id'], product_data)

                        else:
                            if main_session_id in picked_up_objects and product in picked_up_objects[main_session_id]:
                                picked_up_objects[main_session_id][product] += -1
                                picked_up_objects[inner_data_values['session_id']]['conf'] = 100

                            # product_data = load_product_data(r_products, product)
                            # product_data = {"id":product_data['id']}
                            # send_pickup_event('drop', main_session_id, product_data)

                    # For weight sensor based
                    else:
                        product = inner_data_values['product']
                        if inner_data_values['action_type'] == "picked":
                            if inner_data_values['session_id'] in picked_up_objects and product in picked_up_objects[inner_data_values['session_id']]:
                                    picked_up_objects[inner_data_values['session_id']][product] += 1
                                    picked_up_objects[inner_data_values['session_id']]['conf'] = 50
                            else:
                                if inner_data_values['session_id'] not in picked_up_objects:
                                    picked_up_objects[inner_data_values['session_id']] = {}
                                picked_up_objects[inner_data_values['session_id']][product] = 1
                                picked_up_objects[inner_data_values['session_id']]['conf'] = 50

                            # product_data = load_product_data(r_products, product)
                            # send_pickup_event('pickup', inner_data_values['session_id'], product_data)
                            
                        else:
                            if main_session_id in picked_up_objects and product in picked_up_objects[main_session_id]:
                                picked_up_objects[main_session_id][product] += -1
                                picked_up_objects[inner_data_values['session_id']]['conf'] = 50
                    

                            # product_data = load_product_data(r_products, product)
                            # product_data = {"id":product_data['id']}
                            # send_pickup_event('drop', main_session_id, product_data)

                    save_to_json(picked_up_objects, json_file)

        except Exception as e:
            print(e)
            continue




############################### Variable for the define functions ###############################

entry_area = [(210, -4),(250, 272),(130, 324),(58, 12)]
yolo_model_name = "yolov8m-pose.engine"
frame_size = (1080, 600)


# camera urls

entry_cam = "rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=1&subtype=0"
process_cam7 = "rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=7&subtype=0"
process_cam3 = "rtsp://dev:Admin@1234@192.168.1.170:554/cam/realmonitor?channel=3&subtype=0"


# Weight sensor variables

GET_WEIGHT = [0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03]
TARE = [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]

# Specify server host and port
HOST = '192.168.1.118'
PORT = 8235

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(2)
client_socket, addr = server_socket.accept()



############################### Call all function Thread ###############################

# fetching_product_data_thread = threading.Thread(target=fetching_product_data)

entry_thread = threading.Thread(target=entry_camera, args=(entry_cam, yolo_model_name, entry_area, frame_size,))
process_thread1 = threading.Thread(target=process_camera, args=(process_cam7, 7, yolo_model_name, frame_size,))
process_thread2 = threading.Thread(target=process_camera, args=(process_cam3, 3, yolo_model_name, frame_size,))



check_out_thread = threading.Thread(target=checkout_scenario)
weight_thread = threading.Thread(target=handle_weight_changes, args=(client_socket, GET_WEIGHT, TARE,))
decission_thread = threading.Thread(target=decission_maker)



# fetching_product_data_thread.start()

entry_thread.start()
process_thread1.start()
process_thread2.start()
check_out_thread.start()

weight_thread.start()
decission_thread.start()





