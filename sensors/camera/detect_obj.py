from PIL import Image
import numpy as np

# Загрузка изображений и преобразование в ЧБ
img1 = np.array(Image.open("captured_photo_1.jpg").convert('L'))
img2 = np.array(Image.open("captured_photo_2.jpg").convert('L'))


