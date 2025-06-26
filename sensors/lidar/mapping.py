import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# Загрузка карты
map_img = cv2.imread("map_1.png")
if map_img is None:
    raise ValueError("Не удалось загрузить карту")

# Параметры
SCALE = 100  # пикселей на метр
LIDAR_MAX_RANGE = 5.0  # метров
LIDAR_ANGLE_RANGE = 360  # градусов
LIDAR_RESOLUTION = 5  # градусов между лучами

# Начальное положение робота (метры)
robot_x, robot_y = 3.78, 5.07
robot_angle = 0  # радианы

# Создаем бинарную карту (1 - препятствие, 0 - свободно)
binary_map = np.zeros((map_img.shape[0], map_img.shape[1]))
black_pixels = np.all(map_img == [0, 0, 0], axis=-1)
binary_map[black_pixels] = 1


def simulate_lidar(x, y, angle, binary_map, max_range, angle_range=360, resolution=1):
    """Симуляция данных лидара"""
    distances = []
    angles_rad = []

    for angle_deg in np.arange(0, angle_range, resolution):
        ray_angle = angle + math.radians(angle_deg)
        ray_x, ray_y = x * SCALE, y * SCALE
        dx, dy = math.cos(ray_angle), math.sin(ray_angle)

        max_pixels = max_range * SCALE
        distance = max_range

        for step in np.arange(0, max_pixels, 0.5):
            ray_x += dx * 0.5
            ray_y += dy * 0.5

            if (ray_x < 0 or ray_y < 0 or
                    ray_x >= binary_map.shape[1] or
                    ray_y >= binary_map.shape[0]):
                distance = max_range
                break

            if binary_map[int(ray_y), int(ray_x)] == 1:
                distance = step / SCALE
                break

        distances.append(distance)
        angles_rad.append(ray_angle)

    return np.array(distances), np.array(angles_rad)


def draw_lidar_scan(img, distances, angles, robot_x, robot_y, color=(0, 255, 0)):
    """Отрисовка скана лидара на изображении"""
    for dist, angle in zip(distances, angles):
        if dist >= LIDAR_MAX_RANGE:
            continue

        x = int((robot_x + dist * math.cos(angle)) * SCALE)
        y = int((robot_y + dist * math.sin(angle)) * SCALE)

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 2, color, -1)


# Главный цикл
cv2.namedWindow("Robot Navigation")
print("Управление: WASD (движение), Q/E (поворот), ESC (выход)")

while True:
    # Создаем копию карты для отрисовки
    display_img = map_img.copy()

    # Симулируем лидар
    distances, angles = simulate_lidar(robot_x, robot_y, robot_angle,
                                       binary_map, LIDAR_MAX_RANGE,
                                       LIDAR_ANGLE_RANGE, LIDAR_RESOLUTION)

    # Отрисовываем робота (красный круг)
    robot_px = int(robot_x * SCALE)
    robot_py = int(robot_y * SCALE)
    cv2.circle(display_img, (robot_px, robot_py), 8, (0, 0, 255), -1)

    # Отрисовываем направление робота (линия)
    end_x = int((robot_x + 0.2 * math.cos(robot_angle)) * SCALE)
    end_y = int((robot_y + 0.2 * math.sin(robot_angle)) * SCALE)
    cv2.line(display_img, (robot_px, robot_py), (end_x, end_y), (0, 0, 255), 2)

    # Отрисовываем сканы лидара (зеленые точки)
    draw_lidar_scan(display_img, distances, angles, robot_x, robot_y)

    # Показываем изображение
    cv2.imshow("Robot Navigation", display_img)

    # Обработка клавиш
    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break

    # Управление
    move_speed = 0.05
    turn_speed = 0.1

    if key == ord('w'):
        robot_x += move_speed * math.cos(robot_angle)
        robot_y += move_speed * math.sin(robot_angle)
    elif key == ord('s'):
        robot_x -= move_speed * math.cos(robot_angle)
        robot_y -= move_speed * math.sin(robot_angle)
    elif key == ord('a'):
        robot_x += move_speed * math.cos(robot_angle - math.pi / 2)
        robot_y += move_speed * math.sin(robot_angle - math.pi / 2)
    elif key == ord('d'):
        robot_x += move_speed * math.cos(robot_angle + math.pi / 2)
        robot_y += move_speed * math.sin(robot_angle + math.pi / 2)
    elif key == ord('q'):
        robot_angle -= turn_speed
    elif key == ord('e'):
        robot_angle += turn_speed

cv2.destroyAllWindows()