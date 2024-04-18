from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a YOLOv8n model from scratch

model.info()  # display model information
model.train(data="coco128.yaml", epochs=100)  # train the model