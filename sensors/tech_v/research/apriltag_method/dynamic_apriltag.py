import numpy as np
import cv2
import math
import json
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sensors.tech_v.instruments import *

# h = 240
# w = 320
# window = [[360, 160],[360, 480], [360, 480]]
data_2 = []
with open("ancher.json", "rb") as f:
    ancher = json.load(f)
with open("poligon.json", "rb") as f2:
    poligon_center_p = json.load(f2)
logs_file = "logs_no_autofocus/move_logs.txt"
video_file = "logs_no_autofocus/move.webm"

r = 0.35 / 2
tag_ancher = ["35", "36", "37", "32", "33", "34", "24", "25", "26", "27", "29", "18", "19",
              "20", "21", "22"]
poligon = np.array([[-6 * r, 4 * r], [6 * r, 4 * r], [6 * r, -4 * r], [-6 * r, -4 * r], [-6 * r, 4 * r]])

data = {"35": [],
        "36": [],
        "37": [],
        "32": [],
        "33": [],
        "34": [],
        "24": [],
        "25": [],
        "26": [],
        "27": [],
        "29": [],
        "18": [],
        "19": [],
        "20": [],
        "21": [],
        "22": []
        }


def reform(tag, tvec):
    t = time.time()
    if tag == 32:
        return [-tvec[2][0] + ancher[str(tag)][0], tvec[0][0] + ancher[str(tag)][1], t]
    if tag == 33:
        return [-tvec[2][0] + ancher[str(tag)][0], tvec[0][0] + ancher[str(tag)][1], t]
    if tag == 34:
        return [-tvec[2][0] + ancher[str(tag)][0], tvec[0][0] + ancher[str(tag)][1], t]

    if tag == 35:
        return [tvec[2][0] + ancher[str(tag)][0], -tvec[0][0] + ancher[str(tag)][1], t]
    if tag == 36:
        return [tvec[2][0] + ancher[str(tag)][0], -tvec[0][0] + ancher[str(tag)][1], t]
    if tag == 37:
        return [tvec[2][0] + ancher[str(tag)][0], -tvec[0][0] + ancher[str(tag)][1], t]

    if tag == 18:
        return [tvec[0][0] + ancher[str(tag)][0], tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 19:
        return [tvec[0][0] + ancher[str(tag)][0], tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 20:
        return [tvec[0][0] + ancher[str(tag)][0], tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 21:
        return [tvec[0][0] + ancher[str(tag)][0], tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 22:
        return [tvec[0][0] + ancher[str(tag)][0], tvec[2][0] + ancher[str(tag)][1], t]

    if tag == 24:
        return [-tvec[0][0] + ancher[str(tag)][0], -tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 25:
        return [-tvec[0][0] + ancher[str(tag)][0], -tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 26:
        return [-tvec[0][0] + ancher[str(tag)][0], -tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 27:
        return [-tvec[0][0] + ancher[str(tag)][0], -tvec[2][0] + ancher[str(tag)][1], t]
    if tag == 29:
        return [-tvec[0][0] + ancher[str(tag)][0], -tvec[2][0] + ancher[str(tag)][1], t]


def main(frame):
    camera_matrix = np.array([
        [499.88507389877321, 0, 323.12957997277294],
        [0, 497.29148964227164, 246.47731222696228],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([[
        -8.6670522909409886, 16.592497657565755,
        0.0081155977353221925, 0.0011131943241077956, 14.469237940421825,
        -8.6145179947992574, 16.082698259100866, 15.756075391966851, 0.,
        0., 0., 0., 0., 0.
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

    # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # # x, y, w, h = 120, 160, 420, 300
    # x, y, w, h = 160, 160, 320, 270
    # cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    #
    # frame_new = cv2.bitwise_and(frame, frame, mask=mask)

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            tag_id = ids[i][0]
            # if tag_id != 32:
            #     continue
            # Угловые точки
            corner_points = corners[i].astype(np.float32)

            # Решение PnP задачи для определения позы
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corner_points[0],  # Убираем лишнюю размерность
                camera_matrix,
                dist_coeffs
            )

            if success:
                # data[str(tag_id)].append([tvec[0][0], tvec[1][0], tvec[2][0]])
                data[str(tag_id)].append(reform(tag_id, tvec))
                data_2.append(reform(tag_id, tvec))
                # print(data)
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
                             f"X: {-tvec[0][0]:.2f}m, Y: {tvec[1][0]:.2f}m, Z: {tvec[2][0]:.2f}m",

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

            cv2.imshow('Dynamic apriltag', frame)


if __name__ == "__main__":
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        main(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    traj_true = np.array(
        [[-5 * r, -3 * r], [-3 * r, -3 * r], [-r, -3 * r], [r, -3 * r], [3 * r, -3 * r], [5 * r, -3 * r]])
    dyn = read_txt(logs_file)

    fig, ax = plt.subplots()
    ax.set_title("ТЭГ 34")
    ax.plot(poligon[:, 0], poligon[:, 1], "-o", color="black")
    for tag in tag_ancher:
        ax.scatter(ancher[tag][0], ancher[tag][1], color="red")
        ax.text(ancher[tag][0], ancher[tag][1], tag)
    for i in range(1, 25):
        c = (poligon_center_p[str(i)][0] * r, poligon_center_p[str(i)][1] * r)
        circle = Circle(c, r, fill=False)
        ax.add_patch(circle)
        ax.scatter(poligon_center_p[str(i)][0] * r, poligon_center_p[str(i)][1] * r, color="black")
    ax.plot(dyn['x'], dyn['y'], 'b-', marker='o', markersize=3)
    ax.plot(traj_true[:, 0], traj_true[:, 1])
    # ax.plot(-np.array(data["34"])[:, 2] + 1.05, np.array(data["34"])[:, 0] - 0.35, "-o")
    for tag_i in tag_ancher:
        if len(data[tag_i]) == 0:
            continue
        ax.plot(np.array(data[tag_i])[:, 0], np.array(data[tag_i])[:, 1], "-o")
    # ax.plot(np.array(data["34"])[:, 0], np.array(data["34"])[:, 1], "-o")
    ax.grid()

    plt.figure(2)
    plt.plot(np.array(data_2)[:, 2], np.array(data_2)[:, 0])
    plt.plot(dyn['x'], 'b-', marker='o', markersize=3)
    plt.show()
