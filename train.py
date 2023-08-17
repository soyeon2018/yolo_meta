import torch
from ultralytics import YOLO
import ultralytics

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print('...........................')
    print(ultralytics.checks())

    model = YOLO('yolov8n.pt')
    model.train(data='Fish-44/data.yaml', imgsz=640, epochs=10, batch=4)

    # gpu 사용할 시
    # model.train(data='Fish-44/data.yaml', imgsz=640, epochs=10, batch=4, device=0 (or device=[0,1]))