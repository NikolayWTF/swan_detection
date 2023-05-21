import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np

answer_file = open("submission.txt", "w")
answer_file.write("name;class\n")
model = YOLO("yolov8n.pt")
model_swan = YOLO("best.pt")
test_path = "dataset"
photos = os.listdir(test_path)
classes_name = ["шипун", "кликун", "малый"]

for file_name in photos:
    path_to_file_name = test_path + '/' + file_name

    results = model(path_to_file_name,
                    conf=0.5, vid_stride=True, device="mps", task='detect', stream=True)

    classes = [0, 0, 0]
    result = -1
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

                # Массив с confidence
                res_prob = result[0].probs
                max_val = max(res_prob)
                i = 0
                while i < 3:
                    if (res_prob[i] == max_val):
                        ind = i
                    i += 1
                classes[ind] += max_val
    ans = file_name
    if result == -1:
        ans += ";1\n"

    else:
        max_val = max(classes)
        ind = classes.index(max_val)
        swan = result[0].names[ind]

        if swan == "shipun":
            ans += ";3\n"
        else:
            if swan == "klikun":
                ans += ";2\n"
            else:
                ans += ";1\n"
        answer_file.write(ans)

answer_file.close()