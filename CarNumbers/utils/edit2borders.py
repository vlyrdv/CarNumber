import cv2
import numpy as np
from ultralytics import YOLO
from CarNumbers.utils.data import two_array, three_array

model = YOLO("CarNumbers/model/detect_model.pt")


def edit_to_borders(image_path, start_image_path):
    mas = []
    background = cv2.imread("CarNumbers/state/background_number.jpg")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 112))
    cv2.imwrite(image_path, img)
    res = model.predict(image_path)
    for result in res:

        boxes = result.boxes
        cunt = 0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # license_plate = img[int(y1):int(y2), int(x1):int(x2)]
            mas.append([x1, y1, x2, y2])


    mas = sorted(mas)
    if len(mas) > 9:
        mas = mas[:9]
    if len(mas) == 8:
        for i in range(len(mas)):
            x1, y1, x2, y2 = mas[i]
            license_plate = img[int(y1):int(y2), int(x1):int(x2)]

            roi = background[two_array[i][1]:two_array[i][3], two_array[i][0]:two_array[i][2]]

            img_resized = cv2.resize(license_plate, (two_array[i][2] - two_array[i][0], two_array[i][3] - two_array[i][1]))

            roi[:] = img_resized
        cv2.imwrite(f"CarNumbers/output/res-{start_image_path}", background)


    elif len(mas) == 9:
        for i in range(len(mas)):
            x1, y1, x2, y2 = mas[i]
            license_plate = img[int(y1):int(y2), int(x1):int(x2)]

            roi = background[three_array[i][1]:three_array[i][3], three_array[i][0]:three_array[i][2]]

            img_resized = cv2.resize(license_plate, (three_array[i][2] - three_array[i][0], three_array[i][3] - three_array[i][1]))

            roi[:] = img_resized
        cv2.imwrite(f"CarNumbers/output/res-{start_image_path}", background)
    else:
        cv2.imwrite(f"CarNumbers/output/res-{start_image_path}", img)



