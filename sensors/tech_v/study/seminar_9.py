import cv2
import numpy as np
import time
import json

# --- Калибровка камеры ---
""" Задание 1 Впишите свою матрицу внутренних параметров камеры"""

camera_matrix = np.array([
    # [655.11500748680783, 0., 326.34508380835172],
    # [0., 650.31330490929088, 204.26521995888933],
    # [0., 0., 1.]
])

"""Введите свои коэффициенты дисторсии"""
dist_coeffs = np.array([[
    # -0.016959543480066588, 1.8942677470603377,
    # 0.049221312557086207, 0.0084370752827089609,
    # 6.0964415633256692, 0.3192082531679839,
    # 0.077599179103162622, 11.69222684461638,
    # 0., 0., 0., 0., 0., 0.
]])


# --- Преобразование матрицы вращения в углы Эйлера ---
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([z, y, x])  # yaw, pitch, roll


def main():
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: камера недоступна")
        return

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)


    """
    Задание 2
    Пропишите обработчик первого кадра: 
                1) корректировка согласно данным калибровки, 
                2) преобразование в оттенки серого
                3) Увеличение резкости с помощью алгоритма CLAHE
                4) Фильтрация изображения 
    """

    # frame_undistorted =
    # gray =
    # clahe =
    # gray =
    # gray =

    kp1, des1 = sift.detectAndCompute(gray, None)

    R_total = np.eye(3)
    yaw_history = []

    # Инициализация предыдущих значений для фильтрации
    yaw_prev = 0.0
    pitch_prev = 0.0
    roll_prev = 0.0
    alpha_smooth = 0.3  # коэффициент экспоненциального сглаживания
    max_delta = 10.0  # макс. изменение угла за кадр

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        kp2, des2 = sift.detectAndCompute(gray, None)

        if des2 is None or des1 is None or len(kp2) < 10:
            cv2.imshow("Camera", frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        """ 
        Задание 3 Сопоставьте изображения и выберите самые лучшие совпадения
        """
        #matches =
        #good =

        if len(good) > 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

            """Задание 4 Определите внутренюю матрицу"""
            # E, mask =

            if E is not None:
                _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, camera_matrix)
                inliers_ratio = np.sum(mask_pose) / len(mask_pose)

                # Обновляем матрицу только для стабильных кадров
                if inliers_ratio >= 0.6:
                    R_total = R @ R_total

                    euler_angles = rotation_matrix_to_euler_angles(R_total)
                    yaw, pitch, roll = np.degrees(euler_angles)

                    # Ограничение скачков
                    if abs(yaw - yaw_prev) > max_delta:
                        yaw = yaw_prev
                    if abs(pitch - pitch_prev) > max_delta:
                        pitch = pitch_prev
                    if abs(roll - roll_prev) > max_delta:
                        roll = roll_prev

                    # Экспоненциальное сглаживание
                    yaw = alpha_smooth * yaw + (1 - alpha_smooth) * yaw_prev
                    pitch = alpha_smooth * pitch + (1 - alpha_smooth) * pitch_prev
                    roll = alpha_smooth * roll + (1 - alpha_smooth) * roll_prev

                    yaw_prev, pitch_prev, roll_prev = yaw, pitch, roll

                    yaw_history.append({
                        "time": time.time(),
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "roll_deg": roll
                    })

                    """Задание 5 Раскомментируйте необходимый угол"""
                    # cv2.putText(frame_undistorted, f"Yaw: {yaw:.1f} deg", (10, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # cv2.putText(frame_undistorted, f"Pitch: {pitch:.1f}", (10, 60),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # cv2.putText(frame_undistorted, f"Roll: {roll:.1f}", (10, 90),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera", frame_undistorted)
        kp1, des1 = kp2, des2

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    with open("vo_.json", "w", encoding="utf-8") as f:
        json.dump(yaw_history, f, ensure_ascii=False, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    print("Данные курса сохранены в yaw_data_dynamics.json")


if __name__ == "__main__":
    main()
