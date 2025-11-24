import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import cv2
import numpy as np


def read_txt(filename):
    parsed_data = []
    with open(filename, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            parsed_data.append(record)
    df = pd.DataFrame(parsed_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].iloc[0]
    df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
    return df


# dyn = read_txt('dynamic_s12_2.txt')


def detect_apriltag_opencv():
    # Создаем детектор для новых версий OpenCV
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # Инициализация видеопотока
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Ошибка: Не удалось подключиться к камере")
        return

    print("Обнаружение AprilTag (tag36h11) с помощью OpenCV")
    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение AprilTag - ПРАВИЛЬНЫЙ ВЫЗОВ ДЛЯ ArucoDetector
        corners, ids, rejected = detector.detectMarkers(frame)

        # Если найдены теги
        if ids is not None:
            # Рисование обнаруженных тегов
            for i in range(len(ids)):
                # Угловые точки (преобразуем в целые числа)
                corner_points = corners[i].astype(int)

                # Рисование контура тега
                cv2.polylines(frame, [corner_points], True, (0, 255, 0), 3)

                # Рисование центра тега
                center = corner_points[0].mean(axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

                # Отображение ID тега
                tag_id = ids[i][0]
                cv2.putText(frame, f"ID: {tag_id}",
                            tuple(corner_points[0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                print(f"Обнаружен тег ID: {tag_id}")

        # Отображение количества обнаруженных тегов
        count = 0 if ids is None else len(ids)
        cv2.putText(frame, f"Тегов обнаружено: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('AprilTag Detection - OpenCV', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_apriltag_opencv()
