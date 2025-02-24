if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO(r'runs\detect\train4\weights\best.pt')

    model.val()