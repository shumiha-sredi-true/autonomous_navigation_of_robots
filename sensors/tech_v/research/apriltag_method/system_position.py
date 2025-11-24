import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Circle
from sensors.tech_v.instruments import *

f = 24.948
dt = 1 / f
file_name = "logs_no_autofocus/move_up.webm"
r = 0.35 / 2
poligon = np.array([[-6 * r, 4 * r], [6 * r, 4 * r], [6 * r, -4 * r], [-6 * r, -4 * r], [-6 * r, 4 * r]])
with open("poligon.json", "rb") as f2:
    poligon_center_p = json.load(f2)
CALIBRATION_TAG_ID = 8
data = {"tag": 14,
        "xy": []}


def main(file):
    camera_matrix = np.array([[1.44464602e03, 0, 9.06452518e02],
                              [0, 1.43108865e03, 4.68944978e02],
                              [0, 0, 1]])
    dist_coeffs = np.array([[-0.172642475, -0.440297855, 1.11232097e-04, 5.56096772e-04, 0.504451308]])
    TAG_SIZE = 0.1
    R_calib_matrix = None
    t0_calib_vector = None

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    cap = cv2.VideoCapture(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        if ids is None:
            continue

        for i, tag_id_array in enumerate(ids):
            tag_id = int(tag_id_array[0])
            if tag_id != 14 and tag_id != CALIBRATION_TAG_ID:
                continue
            marker_corners = corners[i][0]
            obj_points = np.array([[-TAG_SIZE / 2, TAG_SIZE / 2, 0],
                                   [TAG_SIZE / 2, TAG_SIZE / 2, 0],
                                   [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                                   [-TAG_SIZE / 2, -TAG_SIZE / 2, 0]], dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners.astype(np.float32), camera_matrix,
                                               dist_coeffs)
            if not success:
                continue

            tvec_flat = tvec.flatten()
            if tag_id == CALIBRATION_TAG_ID and (R_calib_matrix is None or t0_calib_vector is None):
                R_temp, _ = cv2.Rodrigues(rvec)
                R_calib_matrix = R_temp
                t0_calib_vector = tvec_flat
                continue

            if R_calib_matrix is not None and t0_calib_vector is not None:
                tvec_calib = R_calib_matrix.T @ (tvec_flat - t0_calib_vector)/1.2
                if tag_id == 14:
                    data["xy"].append([tvec_calib[0], tvec_calib[1]])
                now = time.time()

        cv2.imshow('AprilTag Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(file_name)
    logs_file = "logs_no_autofocus/move_logs.txt"
    dyn = read_txt(logs_file)
    fig, ax = plt.subplots()
    ax.set_title("ТЭГ 34")
    ax.plot(poligon[:, 0], poligon[:, 1], "-o", color="black")
    ax.plot(np.array(data["xy"])[:, 0], np.array(data["xy"])[:, 1])
    traj_true = np.array(
        [[-5 * r, -3 * r], [-3 * r, -3 * r], [-r, -3 * r], [r, -3 * r], [3 * r, -3 * r], [5 * r, -3 * r]])
    for i in range(1, 25):
        c = (poligon_center_p[str(i)][0] * r, poligon_center_p[str(i)][1] * r)
        circle = Circle(c, r, fill=False)
        ax.add_patch(circle)
        ax.scatter(poligon_center_p[str(i)][0] * r, poligon_center_p[str(i)][1] * r, color="black")
    ax.plot(dyn['x'], dyn['y'], 'b-', marker='o', markersize=3)
    # ax.plot(traj_true[:, 0], traj_true[:, 1])
    ax.grid()
    plt.show()

