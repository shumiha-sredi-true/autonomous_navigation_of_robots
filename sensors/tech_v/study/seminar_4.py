import cv2

image = cv2.imread('photos/scale_end.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("original",image)


# Изменение контраста (умножение на коэффициент)
contrast = cv2.multiply(image, 1.5)
cv2.imshow("contrast", contrast)

inverted = 255 - image
cv2.imshow("Inversion", inverted)
inverted_rgb = cv2.bitwise_not(image)# для цветного изображения

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binarization", binary)

# Обрезка (y:y+h, x:x+w)
cropped = image[50:200, 100:300]
cv2.imshow("Cut", cropped)

# Изменение размера
resized = cv2.resize(image, (400, 300))
cv2.imshow("Resized", resized)


image_g = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow("GaussianBlur", image_g)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image)
cv2.imshow("CLAHE", image_clahe)

cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)

# Освобождаем камеру и закрываем окна
cap.release()
cv2.destroyAllWindows()



