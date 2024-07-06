from ultralytics import YOLO
import numpy as np
import cv2
from CarNumbers.utils.rotate_image import rotated_img
from CarNumbers.utils.improvement import start_improvment
from CarNumbers.utils.edit2borders import edit_to_borders
import time

class EditImage:
    def __init__(self):
        self.model = YOLO("CarNumbers/model/seg_model.pt")

    def quality_improvement(self):
        start_improvment()


    def draw_normal_img(self, image_path, max_power=False):
        self.image = image_path
        img = cv2.imread(self.image)
        img = cv2.resize(img, (512, 112))
        cv2.imwrite(self.image, img)
        max_power = max_power

        result = self.model.predict(self.image)


        img = np.copy(result[0].orig_img)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        contour = result[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
        # print(res[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2))
        bin_image = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        itog_image = rotated_img(self.image, bin_image)




        if max_power:

            cv2.imwrite(f"CarNumbers/utils/Real-ESRGAN/inputs/start.jpg", itog_image)

            self.quality_improvement()

            itog_image = cv2.imread("CarNumbers/utils/Real-ESRGAN/results/start_out.jpg")

            cv2.imwrite(f"CarNumbers/state/intermediate_storage/itog_{self.image}", itog_image)
        else:
            cv2.imwrite(f"CarNumbers/state/intermediate_storage/itog_{self.image}", itog_image)

    def edit_form(self):
        edit_to_borders(f"CarNumbers/state/intermediate_storage/itog_{self.image}", self.image)










# start = time.time()
# model = EditImage()
# model.draw_normal_img("IMAGE-PATH", max_power=True)
# model.edit_form()
# print(time.time() - start)
