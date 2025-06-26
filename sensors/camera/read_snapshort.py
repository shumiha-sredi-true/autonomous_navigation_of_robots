import cv2

cap = cv2.VideoCapture(1)

# Запуск
if not cap.isOpened():
    print("Ошибка: Камера не найдена или не доступна!")
    exit()
print("Камера успешно запущена. Нажмите 'q', чтобы выйти.")

k = 1
# Захват кадра
while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр!")
        break
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('captured_photo_'+str(k)+'.jpg', frame)
        k += 1
        print("Фото сохранено!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
