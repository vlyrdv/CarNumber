import cv2
import numpy as np


def rotated_img(link_photo, seg_img):
    def sort_corners(corners):
        center = np.mean(corners, axis=0)

        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])

        corners_sorted = sorted(corners, key=angle_from_center)

        top_left_index = np.argmin([np.linalg.norm(corner) for corner in corners_sorted])
        corners_sorted = corners_sorted[top_left_index:] + corners_sorted[:top_left_index]

        return corners_sorted


    def take_cord(seg_photo):
        image = seg_photo

        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        sorted_corners = sort_corners(box.tolist())
        list_cord = []
        for point in sorted_corners:
            list_cord.append(list(point))

        return list_cord

    list_cord = take_cord(seg_img)
    image = cv2.resize(cv2.imread(link_photo), (512, 112))

    pts1 = np.float32(list_cord)
    width, height = 512, 112
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(image, matrix, (width, height))
    border_width = 10
    border_color = (255, 255, 255)
    img_padded = cv2.copyMakeBorder(result, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT,
                                    value=border_color)

    img_padded = cv2.copyMakeBorder(result, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=border_color)

    img_padded = cv2.resize(img_padded, (512, 112))

    return img_padded


# link_photo = 'Unknown-10.png'
# seg_photo = 'Unknown-9.png'
#
# cv2.imshow('', rotated_img(link_photo, seg_photo))
# cv2.waitKey(0)