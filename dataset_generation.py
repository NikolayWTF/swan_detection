import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

model = YOLO("yolov8n.pt")
test_path = "klikun"
photos = os.listdir(test_path)

photo_num = 1
for file_name in photos:
    path_to_file_name = test_path + '/' + file_name

    results = model(path_to_file_name,
                    conf=0.7, vid_stride=True, device="mps", task='detect', stream=True)

    for r in results:

        boxes = r.boxes
        for box in boxes:
            if model.names[int(box.cls)] == "bird":
                x_0 = int(box.xyxy[0][0])
                y_0 = int(box.xyxy[0][1])
                x_1 = int(box.xyxy[0][2])
                y_1 = int(box.xyxy[0][3])
                img = r.orig_img[y_0:y_1, x_0:x_1]
                cv2.imwrite("dataset/klikun/" + str(photo_num) + ".jpg", img)
                photo_num += 1





