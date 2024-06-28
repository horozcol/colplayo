from ultralytics import YOLO
import cv2
from util import get_car, read_license_plate
from colorama import Fore, Back, Style
import os

os.environ["XDG_SESSION_TYPE"] = "xcb"
#dictionary
results = {}
# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
# class we are looking for
vehicles = [2, 3, 4, 5]
# load source
cam = 0

#cap = cv2.VideoCapture('./centro2.mp4')
cap = cv2.VideoCapture(0)


# read frames
ret = True
while ret:

    ret, frame = cap.read()

    if ret:
    # detect vehicles
        detections = coco_model(frame)[0]
    # if the class id we looking for is in the actual detection, append to the variable called detections_
        detections_ = []


        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                #cv2.imshow('test', frame)

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            car_id = 1

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 80, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow('original', license_plate_crop)
                cv2.imshow('threshold', license_plate_crop_thresh)


                # read license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:

                    results[car_id] = {'license_plate': {'bbox': [x1,y1,x2,y2],
                                                         'text': license_plate_text,
                                                         'bbox_score': score,
                                                         'text_score': license_plate_text_score}}
                    print(Back.YELLOW + f'\n \n Placa detectada: {license_plate_text}\n score: {100*score: .2f} % \n text score: {100*license_plate_text_score: .2f} %' )
                    print(Back.RESET)
    # cv2.waitKey(1000)
    #
    # if cam == 0:
    #     cam = 2
    #
    # else:
    #     cam =0




