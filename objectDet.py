from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
model = YOLO('./yolov8n.pt')

while cap.isOpened():
    status, frame = cap.read()

    if not status:
        break

    results = model(frame)
    frame = results[0].plot()
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
