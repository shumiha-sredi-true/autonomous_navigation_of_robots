import cv2
import numpy as np
import math

rob = 2
video_file = ("statics_rob_" + str(rob) + ".mp4")
image_path = "statics_rob_" + str(rob) + ".jpg"


def main(frame):
    camera_matrix = np.array([
        [505.11500748680783, 0, 196],  # fx, 0, cx
        [0, 500.31330490929088, 254],  # 0, fy, cy
        [0, 0, 1]  # 0, 0, 1
    ], dtype=np.float32)

    # camera_matrix = np.array([
    #     [505.11500748680783, 0, 336],  # fx, 0, cx
    #     [0, 500.31330490929088, 204],  # 0, fy, cy
    #     [0, 0, 1]  # 0, 0, 1
    # ], dtype=np.float32)

    dist_coeffs = np.array([[
        -0.016959543480066588, 1.8942677470603377,
        0.049221312557086207, 0.0084370752827089609,
        6.0964415633256692, 0.3192082531679839,
        0.077599179103162622, 11.69222684461638,
        0., 0., 0., 0., 0., 0.
    ]])

    tag_size = 0.1

    # 3D точки углов AprilTag в системе координат тега
    obj_points = np.array([
        [-tag_size / 2, -tag_size / 2, 0],  # Левый верхний
        [tag_size / 2, -tag_size / 2, 0],  # Правый верхний
        [tag_size / 2, tag_size / 2, 0],  # Правый нижний
        [-tag_size / 2, tag_size / 2, 0]  # Левый нижний
    ], dtype=np.float32)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            # Угловые точки
            corner_points = corners[i].astype(np.float32)

            # Решение PnP задачи для определения позы
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corner_points[0],  # Убираем лишнюю размерность
                camera_matrix,
                dist_coeffs
            )
            tag_id = ids[i][0]
            if success:
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

                # Преобразование в градусы
                pitch = math.degrees(x)
                yaw = math.degrees(y)
                roll = math.degrees(z)

                # Отрисовка осей координат (длина 0.05 метра)
                axis_length = 0.05
                axis_points = np.array([
                    [0, 0, 0],
                    [axis_length, 0, 0],  # Ось X (красная)
                    [0, axis_length, 0],  # Ось Y (зеленая)
                    [0, 0, axis_length]  # Ось Z (синяя)
                ], dtype=np.float32)

                # Проецирование 3D осей на 2D изображение
                img_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, camera_matrix, dist_coeffs
                )
                img_points = img_points.reshape(-1, 2).astype(int)

                # Рисование осей
                origin = tuple(img_points[0])
                cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)  # X - красный
                cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)  # Y - зеленый
                cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)  # Z - синий

                # Отображение информации
                info_text = [f"Tag ID: {tag_id}",
                             f"Distance: {distance:.2f}m",
                             f"X: {tvec[0][0]:.2f}m, Y: {tvec[1][0]:.2f}m, Z: {tvec[2][0]:.2f}m",

                             f"Yaw: {yaw:.1f}deg, Pitch: {pitch:.1f}deg, Roll: {roll:.1f}deg"
                             ]
            cv2.line(frame, tuple(corner_points[0, 0].astype(int)), tuple(corner_points[0, 1].astype(int)),
                     (52, 14, 255), 3)
            cv2.line(frame, tuple(corner_points[0, 1].astype(int)), tuple(corner_points[0, 2].astype(int)),
                     (52, 14, 255), 3)
            cv2.line(frame, tuple(corner_points[0, 2].astype(int)), tuple(corner_points[0, 3].astype(int)),
                     (52, 14, 255), 3)
            cv2.line(frame, tuple(corner_points[0, 3].astype(int)), tuple(corner_points[0, 0].astype(int)),
                     (52, 14, 255), 3)

            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 14, 255), 2)
                y_offset += 20


if __name__ == "__main__":
    tag_2 = np.array([1.0525, 0, 0])

    if rob == 1:
        pos_robot_nav_sys = np.array([0.175, -0.175, 0])
        pos_robot_tag = np.array([0.175, 0, 0.175 * 5])
    elif rob == 2:
        pos_robot_nav_sys = np.array([-0.175, -0.175 * 3, 0])
        pos_robot_tag = np.array([3 * 0.175, 0, 0.175 * 7])
    elif rob == 3:
        pos_robot_nav_sys = np.array([-3 * 0.175, 3 * 0.175, 0])
        pos_robot_tag = np.array([-3 * 0.175, 0, 0.175 * 9])

    pos = cv2.imread(image_path)
    text_1 = f"TRUE_rob: X: {pos_robot_tag[0]:.2f}m, Z: {pos_robot_tag[2]:.2f}m"
    text_2 = f"TRUE_nav: X: {pos_robot_nav_sys[0]:.2f}m, Y: {pos_robot_nav_sys[1]:.2f}m"
    cv2.putText(pos, text_1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 14, 255), 2)
    cv2.putText(pos, text_2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 14, 255), 2)
    cv2.imshow("POS ROBOT", pos)

    cap = cv2.VideoCapture(video_file)
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        main(frame)
        cv2.imshow('AprilTag', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
