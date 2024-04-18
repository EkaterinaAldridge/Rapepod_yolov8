from ultralytics import YOLO

model = YOLO('yolov8m-seg.yaml')
model = YOLO('yolov8m-seg.pt')

result = model.train(data='D:\\rapepod_yolov8n\\rapedata.yaml', epochs=100, imgsz=640)
