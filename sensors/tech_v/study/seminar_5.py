import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Загружаем изображение
image_bad_light = cv2.imread('photos/bad_light.jpg')
image_good_light = cv2.imread('photos/good_light.jpg')
image_fog = cv2.imread('photos/fog.jpg')

# Применение усредняющего фильтра
kernel_size = 3  # величина окна
image_bad_light_blur = cv2.blur(image_bad_light, (kernel_size, kernel_size))
image_good_light_blur = cv2.blur(image_good_light, (kernel_size, kernel_size))
image_fog_blur = cv2.blur(image_fog, (kernel_size, kernel_size))

# Визуализация всех изображений
plt.figure(1)
plt.imshow(image_bad_light_blur)
plt.axis('off')

plt.figure(2)
plt.imshow(image_bad_light)
plt.axis('off')

plt.figure(3)
plt.imshow(image_good_light_blur)
plt.axis('off')

plt.figure(4)
plt.imshow(image_good_light)
plt.axis('off')

plt.figure(5)
plt.imshow(image_fog_blur)
plt.axis('off')

plt.figure(6)
plt.imshow(image_fog)
plt.axis('off')

plt.show()

# Фильтр на основе ядра яркости
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

image_bad_light_fk = cv2.filter2D(image_bad_light, -1, kernel)  # -1 означает, что глубина выходного изображения = глубине
# входного
image_good_light_fk = cv2.filter2D(image_good_light, -1, kernel)  # -1 означает, что глубина выходного изображения = глубине
image_fog_fk = cv2.filter2D(image_fog, -1, kernel)  # -1 означает, что глубина выходного изображения = глубине

plt.figure(1)
plt.imshow(image_bad_light_fk)
plt.axis('off')

plt.figure(2)
plt.imshow(image_bad_light)
plt.axis('off')

plt.show()

# Алгоритм CLAHE

# Конвертация в LAB color space
lab = cv2.cvtColor(image_fog, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Применение CLAHE к L-каналу (яркость)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Объединение каналов и конвертация обратно в BGR
limg = cv2.merge((cl, a, b))
result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

plt.figure(1)
plt.imshow(result)
plt.axis('off')

plt.figure(2)
plt.imshow(image_fog_fk)
plt.axis('off')

plt.show()
