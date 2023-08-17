from ultralytics import YOLO

model = YOLO('runs/detect/train2/weights/best.pt')
results = model.predict(source='fish.jpg', show=False, save=True)  # 결과가 리스트

for result in results:
    boxes = result.boxes
    print(boxes)

print(boxes.xywh)  # 바운딩 박스의 좌표
print(boxes.cls)  # 클래스 이름