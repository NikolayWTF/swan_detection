import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
model_swan = YOLO("best.pt")
test_path = "dataset/klikun"
photos = os.listdir(test_path)
classes_name = ["кликун", "малый", "шипун"]
answer = [0, 0, 0]
for file_name in photos:
    path_to_file_name = test_path + '/' + file_name

    results = model(path_to_file_name,
                    conf=0.5, vid_stride=True, device="mps", task='detect', stream=True)

    classes = [0, 0, 0]

    for r in results:

        boxes = r.boxes
        for box in boxes:
            if model.names[int(box.cls)] == "bird":
                x_0 = int(box.xyxy[0][0])
                y_0 = int(box.xyxy[0][1])
                x_1 = int(box.xyxy[0][2])
                y_1 = int(box.xyxy[0][3])
                img = r.orig_img[y_0:y_1, x_0:x_1]
                cv2.imwrite("tmp.jpg", img)
                result = model_swan("tmp.jpg")
                print(result)
                res_prob = result[0].probs
                max_val = max(res_prob)
                i = 0
                while i < 3:
                    if (res_prob[i] == max_val):
                        ind = i
                    i += 1
                classes[ind] += max_val
    max_val = max(classes)
    ind = classes.index(max_val)

    print(classes_name[ind])
    answer[ind] += 1
print(answer)