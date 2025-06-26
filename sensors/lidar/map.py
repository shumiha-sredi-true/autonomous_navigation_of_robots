import cv2
import numpy as np
import matplotlib.pyplot as plt


def simulate_lidar(robot_x, robot_y, map_binary, max_range, angle_range=360, resolution=1):
    """
    Симуляция данных лидара
    :param robot_x: координата x робота в метрах
    :param robot_y: координата y робота в метрах
    :param map_binary: бинарная карта (1 - препятствие, 0 - свободно)
    :param max_range: максимальная дальность лидара в метрах
    :param angle_range: угловой диапазон лидара в градусах
    :param resolution: разрешение лидара в градусах
    :return: массивы расстояний и углов
    """
    height, width = map_binary.shape
    distances = []
    angles = []

    # Масштаб: пиксели в метры (предполагаем 100 пикселей = 1 метр)
    scale = 100.0

    for angle_deg in np.arange(0, angle_range, resolution):
        angle_rad = np.deg2rad(angle_deg)

        # Инициализация луча
        ray_x, ray_y = robot_x * scale, robot_y * scale
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)

        max_pixels = max_range * scale
        distance = max_range

        # Пошаговое продвижение луча
        for step in np.arange(0, max_pixels, 0.5):  # шаг 0.5 пикселя
            ray_x += dx * 0.5
            ray_y += dy * 0.5

            # Проверка выхода за границы карты
            if ray_x < 0 or ray_y < 0 or ray_x >= width or ray_y >= height:
                distance = max_range
                break

            # Проверка столкновения с препятствием
            if map_binary[int(ray_y), int(ray_x)] == 1:
                distance = step / scale
                break

        distances.append(distance)
        angles.append(angle_rad)

    return np.array(distances), np.array(angles)


def lidar_to_pixels(distances, angles, robot_x, robot_y, scale=100.0):
    """
    Преобразует данные лидара в пиксельные координаты
    :param distances: массив расстояний (метры)
    :param angles: массив углов (радианы)
    :param robot_x: координата x робота (метры)
    :param robot_y: координата y робота (метры)
    :param scale: масштаб (пикселей/метр)
    :return: массив пиксельных координат (u, v)
    """
    # Относительные координаты
    x_rel = distances * np.cos(angles)
    y_rel = distances * np.sin(angles)

    # Абсолютные координаты в метрах
    x_abs = robot_x + x_rel
    y_abs = robot_y + y_rel

    # Переводим в пиксели
    u = (x_abs * scale).astype(int)
    v = (y_abs * scale).astype(int)

    return np.column_stack((u, v))


def lidar_simulation(file_image, robot_width, robot_length, f_viz = False):
    image = cv2.imread(file_image)
    if image is None:
        raise ValueError("Не удалось загрузить изображение")

    # Пиксели --> метры
    robot_x = 0.01 * robot_width
    robot_y = 0.01 * robot_length

    # Параметры лидара
    LIDAR_MAX_RANGE = 10.0  # метров
    LIDAR_ANGLE_RANGE = 360  # градусов
    LIDAR_RESOLUTION = 1  # градусов между лучами

    # Создаем бинарную карту препятствий (1 - препятствие, 0 - свободно)
    binary_map = np.zeros((image.shape[0], image.shape[1]))
    black_pixels = np.all(image == [0, 0, 0], axis=-1)
    binary_map[black_pixels] = 1

    # Симулируем данные лидара
    distances, angles = simulate_lidar(robot_x, robot_y, binary_map,
                                       LIDAR_MAX_RANGE,
                                       LIDAR_ANGLE_RANGE,
                                       LIDAR_RESOLUTION)

    # Преобразуем данные лидара в пиксели
    lidar_pixels = lidar_to_pixels(distances, angles, robot_x, robot_y)

    # Создаем изображение для визуализации
    result_img = image.copy()

    # Отмечаем положение робота (красный крест)
    robot_u = int(robot_x * 100)
    robot_v = int(robot_y * 100)
    cv2.drawMarker(result_img, (robot_u, robot_v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # Отмечаем точки лидара (синие кружки)
    for u, v in lidar_pixels:
        if 0 <= u < result_img.shape[1] and 0 <= v < result_img.shape[0]:
            cv2.circle(result_img, (u, v), 2, (255, 0, 0), -1)

    # Сохраняем и показываем результат
    # cv2.imwrite("lidar_result_3.png", result_img)

    if f_viz:
        """
        # Визуализация в полярных координатах
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.scatter(angles, distances, s=1)
        ax.set_title("Данные лидара в полярных координатах", pad=20)
        ax.set_theta_zero_location('E')  # 0 градусов справа
        ax.set_theta_direction(-1)  # По часовой стрелке
        plt.show()

        # Визуализация в декартовых координатах
        cartesian_x = distances * np.cos(angles)
        cartesian_y = distances * np.sin(angles)

        plt.figure(figsize=(10, 10))
        plt.scatter(cartesian_x, cartesian_y, s=1)
        plt.scatter(0, 0, c='r', marker='x', label='Робот')
        plt.xlabel("X (метры)")
        plt.ylabel("Y (метры)")
        plt.title("Данные лидара в декартовых координатах")
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()
        """

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Данные лидара на карте (пиксели)")
        plt.scatter(robot_u, robot_v, c='red', marker='x', label='Робот')
        plt.legend()
        plt.show()

    return lidar_pixels


"""
    Положение робота (в пикселях)
    x y <-> ширина, длина
"""

robot_width = 716
robot_length = 511
lidar_simulation("map_1.png", robot_width, robot_length, f_viz = True)
