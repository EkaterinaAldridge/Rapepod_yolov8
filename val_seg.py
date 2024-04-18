from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
result = model.val(data='coco128-seg.yaml')
print(result)
