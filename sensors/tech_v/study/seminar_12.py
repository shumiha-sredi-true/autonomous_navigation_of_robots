import cv2
import numpy as np


# Добавляем метрическую сетку
def add_metric_grid(image, pixels_per_meter=300):
    h, w = image.shape[:2]

    # Рисуем сетку каждый 0.1 метр (10 см)
    grid_step = 0.1 * pixels_per_meter

    for x in range(0, w, int(grid_step)):
        cv2.line(image, (x, 0), (x, h), (0, 255, 0), 1)

    for y in range(0, h, int(grid_step)):
        cv2.line(image, (0, y), (w, y), (0, 255, 0), 1)

    return image


frame = cv2.imread("photos/photo_0.jpg")

# Исходные точки на изображении (пиксели)
pts_src = np.array([[73, 54], [616, 63], [59, 382], [605, 402]], dtype=np.float32)

# Определяем масштаб: сколько пикселей на метр
# Например, если хотим получить изображение, где 1 метр = 300 пикселей
pixels_per_meter = 300

# Целевые точки в пикселях (преобразуем метры в пиксели)
# Прямоугольник 2.5м x 1.5м
width_px = int(2.5 * pixels_per_meter)
height_px = int(1.5 * pixels_per_meter)

pts_dst_metric = np.array([[-1.25, 0.75], [1.25, 0.75], [-1.25, -0.75], [1.25, -0.75]], dtype=np.float32)

# Масштабируем метры в пиксели
pts_dst = pts_dst_metric * pixels_per_meter + [width_px/2, height_px/2]  # центрируем

# Находим матрицу гомографии
H, status = cv2.findHomography(pts_src, pts_dst)

# Применяем преобразование с правильным размером выхода
warped = cv2.warpPerspective(frame, H, (width_px, height_px))

# Добавляем сетку к результату
warped_with_grid = add_metric_grid(warped.copy())

# Показываем результат
cv2.imshow("Orig", frame)
cv2.imshow("Poligon", warped)
cv2.imshow("Poligon_grid", warped_with_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()