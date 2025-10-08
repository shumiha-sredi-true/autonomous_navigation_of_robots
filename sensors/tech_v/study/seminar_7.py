import cv2

# Загружаем изображение
img_1 = cv2.imread("photos/lighting_start.jpg")
img_2 = cv2.imread("photos/lighting_end.jpg")

gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# --- 1. SIFT ---
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(gray_1, None)
kp2_sift, des2_sift = sift.detectAndCompute(gray_2, None)

img1_sift = cv2.drawKeypoints(img_1, kp1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_sift = cv2.drawKeypoints(img_2, kp2_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- 2. ORB ---
orb = cv2.ORB_create(nfeatures=500)

# Поиск особых точек и вычисление дескрипторов
keypoints_1, descriptors_1 = orb.detectAndCompute(gray_1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(gray_2, None)

# Рисуем найденные точки на изображении
img_keypoints_1 = cv2.drawKeypoints(img_1, keypoints_1, None, color=(0, 255, 0))
img_keypoints_2 = cv2.drawKeypoints(img_2, keypoints_2, None, color=(0, 255, 0))

# Создаём матчинг-дескриптор (Brute Force с Hamming расстоянием)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Сопоставляем дескрипторы
matches = bf.match(descriptors_1, descriptors_2)

# Сортируем по расстоянию (лучшие совпадения первые)
matches = sorted(matches, key=lambda x: x.distance)

# Рисуем первые 50 совпадений
img_matches_orb = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("ORB Matches", img_matches_orb)
cv2.imshow("ORB_1", img_keypoints_1)
cv2.imshow("ORB_2", img_keypoints_2)
cv2.imshow("SIFT_1", img1_sift)
cv2.imshow("SIFT_2", img2_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
