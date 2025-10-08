import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Загрузка изображения
image = mpimg.imread('photos/captured_photo_1.jpg')
# Проверка целостности изображения
"""
    plt.imshow(image)
    plt.axis('off')
    plt.show()
"""
# 2. Определение разрешения изображения и способа хранения изображения
print(image.shape)
# 3. Определение R, G, B одного произвольного пикселя
print(image[300, 100])

# 4. Выделяем три цветовые компоненты
image_r = np.zeros_like(image)
image_g = np.zeros_like(image)
image_b = np.zeros_like(image)

image_r[:, :, 0] = image[:, :, 0]
image_g[:, :, 1] = image[:, :, 1]
image_b[:, :, 2] = image[:, :, 2]
"""
    plt.figure(1)
    plt.imshow(image_r)
    plt.axis('off')
    
    plt.figure(2)
    plt.imshow(image_g)
    plt.axis('off')
    
    plt.figure(3)
    plt.imshow(image_b)
    plt.axis('off')
    plt.show()
"""

# 5. Видео
"""
    plt.ion()
    for i in range(1, 5):
        plt.imshow(mpimg.imread('captured_photo_' + str(i) + '.jpg'))
        plt.axis('off')
        plt.draw()
        plt.pause(1.0)
"""

# Самостоятельная работа 3
# Задание 1
image_traj = mpimg.imread('traj.jpg')
print(image_traj.shape)
# Маска
black_pixels_mask = (image_traj == [0, 0, 0]).all(axis=2)
# Выделение линии
y_coords, x_coords = np.where(black_pixels_mask)
# Преобразование координат
x = x_coords * 0.1
y = y_coords * 0.1 * np.cos(np.pi)
ind_min_x = np.argmin(x)
# Построение траектории в декартовой системе координат
"""
    plt.scatter(x - x[ind_min_x], y - y[ind_min_x])
    plt.grid()
    plt.xlabel("X, м")
    plt.ylabel("Y, м")
    plt.show()
"""


# Задание 2
image_rgb = np.array([
    [[255, 0, 0], [0, 255, 0], [150, 255, 47]],
    [[0, 0, 255], [255, 255, 0], [120, 205, 200]],
    [[255, 255, 255], [167, 55, 100], [20, 25, 20]]
])  # 3x3x3

plt.imshow(image_rgb)
plt.axis('off')
plt.show()
