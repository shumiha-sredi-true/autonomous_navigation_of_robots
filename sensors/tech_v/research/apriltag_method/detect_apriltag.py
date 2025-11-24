import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from sensors.tech_v.instruments import *

f = 24.948
dt = 1 / f
file_name = "logs_no_autofocus/move_to_tag.webm"


def main(file):
    camera_matrix = np.array([
        [796.88690565, 0, 334.2251184],
        [0, 500.46157655, 231.07701353],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([[
        -8.32132971e-01, 4.30770037e+00, 8.12046245e-04, - 2.55496285e-02,
        - 1.44802951e+01, 6.81363144e-01, - 5.97150795e+00, 6.02307653e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00
    ]])

    tag_size = 0.1
    obj_points = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, tag_size / 2, 0],
        [-tag_size / 2, tag_size / 2, 0]
    ], dtype=np.float32)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    cap = cv2.VideoCapture(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            for i in range(len(ids)):
                tag_id = str(ids[i][0])
                print(tag_id)
                if tag_id != "34":
                    continue
                corner_points = corners[i].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corner_points[0],
                    camera_matrix,
                    dist_coeffs
                )

                tag_id = str(ids[i][0])

                if success:
                    # Расчет расстояния и углов
                    distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)

                    rmat, _ = cv2.Rodrigues(rvec)
                    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                    singular = sy < 1e-6

                    if not singular:
                        x = math.atan2(rmat[2, 1], rmat[2, 2])
                        y = math.atan2(-rmat[2, 0], sy)
                        z = math.atan2(rmat[1, 0], rmat[0, 0])
                    else:
                        x = math.atan2(-rmat[1, 2], rmat[1, 1])
                        y = math.atan2(-rmat[2, 0], sy)
                        z = 0

                    pitch = math.degrees(x)
                    yaw = math.degrees(y)
                    roll = math.degrees(z)

                    # Отрисовка осей
                    axis_length = 0.05
                    axis_points = np.array([
                        [0, 0, 0],
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length]
                    ], dtype=np.float32)

                    img_points, _ = cv2.projectPoints(
                        axis_points, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    img_points = img_points.reshape(-1, 2).astype(int)

                    origin = tuple(img_points[0])
                    cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)
                    cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)
                    cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)

                    # Отображение информации
                    info_text = [
                        f"Tag ID: {tag_id}",
                        f"Distance: {distance:.2f}m",
                        f"X: {tvec[0][0]:.2f}m, Y: {tvec[1][0]:.2f}m, Z: {tvec[2][0]:.2f}m",
                        f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg, Roll: {roll:.1f}deg"
                    ]

                    y_offset = 30
                    for text in info_text:
                        cv2.putText(frame, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20

        cv2.imshow('AprilTag Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(file_name)
