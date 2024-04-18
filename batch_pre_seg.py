from ultralytics import YOLO
import numpy as np
import cv2
import os

model = YOLO("yolov8n-seg.pt")
imgsPath = "source"
imgsList = os.listdir(imgsPath)
for imgName in imgsList:
    imgFile = os.path.join(imgsPath, imgName)
    model(imgFile, save_txt=True)
