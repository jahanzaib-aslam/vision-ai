# import os
# import time
# import torch
# from ultralytics import YOLO
# import cvzone
# import random
# import asyncio
# import websockets
# import requests
# import cv2



def process_frame(frame, model):
    # results = model.track(frame, persist=True, stream=True)
    results = model(frame, stream=True)
    results = next(results)
    detection_status_check = len(results.boxes.data) > 0
    pose_boxes = results.boxes.data
    pose_keypoints = results.keypoints.xy
    return detection_status_check, pose_boxes, pose_keypoints

