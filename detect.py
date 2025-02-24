from ultralytics import YOLO

model = YOLO(r'runs\detect\train4\weights\best.pt')

model(r'fruits\datasets\images\test', save=True, classes = [0, 1, 2])