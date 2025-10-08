import cv2
import numpy as np
import time


class AprilTagDetector:
    def __init__(self, camera_id=0, dictionary_name='DICT_APRILTAG_36h11'):
        """
        Инициализация детектора AprilTags с использованием ArUco

        Args:
            camera_id: ID камеры (обычно 0 для встроенной камеры)
            dictionary_name: тип словаря маркеров
        """
        self.camera_id = camera_id
        self.cap = None
        self.dictionary = None
        self.parameters = None
        self.detector = None
        self.marker_size = 0.1  # Размер маркера в метрах (для оценки позы)
        self.setup_detector(dictionary_name)

    def setup_detector(self, dictionary_name):
        """Настройка детектора AprilTags через ArUco"""
        # Сопоставление имен словарей с константами OpenCV
        aruco_dicts = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
            'DICT_APRILTAG_16h5': cv2.aruco.DICT_APRILTAG_16h5,
            'DICT_APRILTAG_25h9': cv2.aruco.DICT_APRILTAG_25h9,
            'DICT_APRILTAG_36h10': cv2.aruco.DICT_APRILTAG_36h10,
            'DICT_APRILTAG_36h11': cv2.aruco.DICT_APRILTAG_36h11,
        }

        if dictionary_name in aruco_dicts:
            dictionary_id = aruco_dicts[dictionary_name]
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            print(f"Используется словарь: {dictionary_name}")
        else:
            # Используем стандартный словарь по умолчанию
            self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            print(f"Словарь {dictionary_name} не найден. Используется DICT_6X6_250")

        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def get_camera_matrix(self):
        """Возвращает матрицу камеры (нужно откалибровать для точных измерений)"""
        # Приблизительные параметры камеры (замените на откалиброванные значения)
        width, height = 640, 480
        fx = 800  # Фокусное расстояние по x
        fy = 800  # Фокусное расстояние по y
        cx = width / 2  # Центр по x
        cy = height / 2  # Центр по y

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)

        return camera_matrix

    def get_distortion_coeffs(self):
        """Возвращает коэффициенты дисторсии"""
        # Нулевые коэффициенты (замените на откалиброванные значения)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return dist_coeffs

    def setup_camera(self, width=640, height=480):
        """Настройка камеры"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            # Пробуем другие ID камер
            for cam_id in [1, 2, 3]:
                self.cap = cv2.VideoCapture(cam_id)
                if self.cap.isOpened():
                    self.camera_id = cam_id
                    print(f"Камера найдена на ID: {cam_id}")
                    break

            if not self.cap.isOpened():
                raise Exception("Не удалось открыть камеру")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"Камера {self.camera_id} настроена: {width}x{height}")

    def detect_tags(self, frame):
        """Обнаружение AprilTags в кадре"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            corners, ids, rejected = self.detector.detectMarkers(gray)
            return corners, ids
        except Exception as e:
            print(f"Ошибка при обнаружении маркеров: {e}")
            return None, None

    def estimate_pose(self, corners):
        """Оценка позы маркера относительно камеры"""
        if corners is None or len(corners) == 0:
            return None, None

        camera_matrix = self.get_camera_matrix()
        dist_coeffs = self.get_distortion_coeffs()

        # Оцениваем позу для каждого маркера
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, camera_matrix, dist_coeffs
        )

        return rvecs, tvecs

    def draw_axes(self, frame, corners, rvecs, tvecs):
        """Отрисовка осей координат для каждого маркера"""
        camera_matrix = self.get_camera_matrix()
        dist_coeffs = self.get_distortion_coeffs()

        for i in range(len(rvecs)):
            # Рисуем оси координат (X, Y, Z)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvecs[i], self.marker_size * 0.5)

            # Добавляем информацию о позиции
            if tvecs is not None:
                tvec = tvecs[i][0]
                position_text = f"Pos: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})"

                # Получаем центр маркера для отображения текста
                center = np.mean(corners[i][0], axis=0).astype(int)

                cv2.putText(frame, position_text, (center[0] - 100, center[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def draw_tags(self, frame, corners, ids):
        """Отрисовка обнаруженных тегов на кадре"""
        if ids is not None and len(ids) > 0:
            # Рисуем обнаруженные маркеры
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Добавляем дополнительную информацию для каждого маркера
            for i in range(len(ids)):
                tag_id = ids[i][0]
                corner_points = corners[i][0]

                # Вычисляем центр маркера
                center_x = int(np.mean(corner_points[:, 0]))
                center_y = int(np.mean(corner_points[:, 1]))

                # Отображаем ID маркера
                cv2.putText(frame, f"ID: {tag_id}", (center_x - 30, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Отображаем координаты центра
                cv2.putText(frame, f"({center_x},{center_y})", (center_x - 40, center_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return frame

    def draw_coordinate_system_info(self, frame):
        """Отрисовка информации о системе координат"""
        height, width = frame.shape[:2]

        # Легенда системы координат
        info_text = [
            "Система координат:",
            "X - красная ось (право)",
            "Y - зеленая ось (вниз)",
            "Z - синяя ось (вперед)",
            "Размер осей: 5 см"
        ]

        for i, text in enumerate(info_text):
            y_pos = height - 150 + i * 25
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Основной цикл обнаружения"""
        self.setup_camera()

        print("Запуск обнаружения AprilTags/ArUco маркеров...")
        print("Нажмите 'q' для выхода")
        print("Нажмите 's' для сохранения текущего кадра")
        print("Нажмите 'c' для изменения словаря маркеров")
        print("Нажмите 'm' для изменения размера маркера")

        fps_time = time.time()
        frame_count = 0
        fps = 0

        # Список доступных словарей для переключения
        dictionaries = [
            'DICT_APRILTAG_16h5',
            'DICT_APRILTAG_25h9',
            'DICT_APRILTAG_36h10',
            'DICT_APRILTAG_36h11',
            'DICT_4X4_50',
            'DICT_5X5_50',
            'DICT_6X6_50'
        ]
        current_dict_index = 3  # DICT_APRILTAG_36h11 по умолчанию

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры")
                break

            # Обнаруживаем теги
            corners, ids = self.detect_tags(frame)

            # Отрисовываем результаты
            frame_with_tags = frame.copy()

            if corners is not None and ids is not None:
                # Оцениваем позу и рисуем оси
                rvecs, tvecs = self.estimate_pose(corners)
                if rvecs is not None and tvecs is not None:
                    self.draw_axes(frame_with_tags, corners, rvecs, tvecs)

                # Рисуем маркеры и информацию
                frame_with_tags = self.draw_tags(frame_with_tags, corners, ids)

            # Рисуем информацию о системе координат
            self.draw_coordinate_system_info(frame_with_tags)

            # Отображаем FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # Отображаем информацию
            cv2.putText(frame_with_tags, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            tag_count = len(ids) if ids is not None else 0
            cv2.putText(frame_with_tags, f"Tags: {tag_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            current_dict = dictionaries[current_dict_index]
            cv2.putText(frame_with_tags, f"Dict: {current_dict}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.putText(frame_with_tags, f"Size: {self.marker_size * 100:.1f} cm", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Показываем результат
            cv2.imshow('AprilTag/ArUco Detection with Axes', frame_with_tags)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_tags)
                print(f"Сохранено: {filename}")
            elif key == ord('c'):
                # Переключение словаря
                current_dict_index = (current_dict_index + 1) % len(dictionaries)
                new_dict = dictionaries[current_dict_index]
                self.setup_detector(new_dict)
                print(f"Переключен на словарь: {new_dict}")
            elif key == ord('m'):
                # Изменение размера маркера
                self.marker_size = float(input("Введите размер маркера в метрах (например 0.1): "))
                print(f"Размер маркера установлен: {self.marker_size} м")

        # Очистка
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Основная функция"""
    try:
        detector = AprilTagDetector(camera_id=1, dictionary_name='DICT_APRILTAG_36h11')
        detector.run()
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что камера подключена и OpenCV установлен правильно")


if __name__ == "__main__":
    main()