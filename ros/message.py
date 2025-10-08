# import cv2
# import numpy as np
#
#
# def main(frame):
#     """
#     Обрабатывает кадр, выделяет особые точки и рисует их.
#     frame: np.ndarray (BGR)
#     return: np.ndarray с отмеченными точками
#     """
#
#     # Конвертируем в оттенки серого
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Детектор особых точек (Shi-Tomasi corners)
#     corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
#
#     if corners is not None:
#         corners = np.int0(corners)
#         for i in corners:
#             x, y = i.ravel()
#             cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # зелёные точки
#
#     return frame
import cv2
import numpy as np

k = 0


def sad(first_frame, frame):
    res = np.abs(first_frame - frame)
    return res


def main(frame):
    global k
    if k == 0:
        first_frame = frame
        k = k + 1


    result = sad(first_frame, frame)

    return result


import cv2
import numpy as np

k = 0
first_frame = 0

def sad(first_frame, frame):
    res = cv2.subtract(first_frame, frame)
    return res


def main(frame):
    global k, first_frame
    if k == 0:
        first_frame = frame
        k = k + 1
    result = sad(first_frame, frame)

    return result
