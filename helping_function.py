# import cv2
import numpy as np
import json
# import os
# import time
# import torch
# from ultralytics import YOLO
# import cvzone
import random
# import asyncio
# import websockets
# import requests
# import threading
import pickle
import zlib


def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def capture_weight_area(frame, weight_position, margin=140):
    x, y = weight_position
    crop_x = max(0, x - margin)
    crop_y = max(0, y - margin)
    crop_w = min(frame.shape[1] - crop_x, margin * 2)
    crop_h = min(frame.shape[0] - crop_y, margin * 2)
    hand_area = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    return hand_area, crop_x, crop_y


def generate_random_id():
    return str(random.randint(10000, 99999))



def compress_data(data):
    # Serialize data to bytes using pickle
    serialized_data = pickle.dumps(data)
    # Compress the serialized data
    compressed_data = zlib.compress(serialized_data)
    return compressed_data


def decompress_data(compressed_data):
    # Decompress the data
    serialized_data = zlib.decompress(compressed_data)
    # Deserialize the bytes back to a Python object
    data = pickle.loads(serialized_data)
    return data


def decission_data(r):
    if len(r.keys()) != 0:
        decission_data_dictionary = {}
        for key in r.keys():
            key_str = key.decode('utf-8')
            value = r.get(key_str)
            decompress_data_values = decompress_data(value)
            decission_data_dictionary[key_str] = decompress_data_values
        return decission_data_dictionary
    else: return {}


def delete_keys_with_prefix(r, prefix):
    # Find all keys that match the given prefix before the underscore
    pattern = f"{prefix}*"    
    matching_keys = r.keys(pattern)
    if matching_keys:
        r.delete(*matching_keys)