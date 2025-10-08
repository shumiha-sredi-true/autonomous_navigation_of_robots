import cv2
import numpy as np
import time # Добавлено для задержки между автоматическими захватами

# --- Параметры калибровки ---
CHESSBOARD_SIZE = (7, 9)  # (внутренние углы по ширине, по высоте) для доски 8x10 клеток
SQUARE_SIZE = 0.02        # Размер клетки в метрах (20 мм)
CAMERA_INDEX = 1          # Индекс камеры (0 обычно встроенная, 1+ внешние)
FRAME_WIDTH = 1920        # Желаемая ширина кадра
FRAME_HEIGHT = 1080       # Желаемая высота кадра
MIN_IMAGES_FOR_CALIBRATION = 10 # Минимальное количество кадров для калибровки
TARGET_IMAGES = 20        # Целевое количество кадров для сбора
MIN_SECONDS_BETWEEN_CAPTURES = 10.0 # Минимальное время (в секундах) между авто-захватами

# --- Подготовка данных ---
obj_points = []
img_points = []

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- Инициализация камеры ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть камеру с индексом {CAMERA_INDEX}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Камера инициализирована. Разрешение: {actual_width}x{actual_height}")
print(f"Используется шахматная доска {CHESSBOARD_SIZE[0]+1}x{CHESSBOARD_SIZE[1]+1} клеток.")
print(f"Размер клетки: {SQUARE_SIZE * 1000} мм.")
print("\n--- Автоматический сбор кадров ---")
print(f"Медленно перемещайте шахматную доску перед камерой под разными углами.")
print(f"Кадры будут сохраняться автоматически при обнаружении углов.")
print(f"Нужно собрать {TARGET_IMAGES} кадров.")
print(f"Нажмите 'q', чтобы прервать сбор или выйти.")

images_captured = 0
last_capture_time = 0.0 # Время последнего успешного захвата
calibration_successful = False
camera_matrix = None
dist_coeffs = None

# --- Цикл автоматического сбора кадров ---
while images_captured < TARGET_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр с камеры.")
        time.sleep(0.5)
        continue

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found_corners_on_frame = False
    current_time = time.time() # Получаем текущее время

    if ret_corners:
        found_corners_on_frame = True
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners_refined, ret_corners)

        # --- Логика автоматического захвата ---
        if current_time - last_capture_time >= MIN_SECONDS_BETWEEN_CAPTURES:
            obj_points.append(objp)
            img_points.append(corners_refined)
            images_captured += 1
            last_capture_time = current_time # Обновляем время последнего захвата
            print(f"Кадр {images_captured}/{TARGET_IMAGES} автоматически сохранен.")
            # Визуальная индикация
            cv2.putText(display_frame, "CAPTURED!", (actual_width // 2 - 150, actual_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
            # Показываем кадр с надписью CAPTURED! на короткое время
            cv2.imshow('Camera Calibration - Frame Capture', display_frame)
            cv2.waitKey(500) # Задержка 500 мс после захвата
            # Сразу переходим к следующей итерации, чтобы не показывать кадр без надписи "CAPTURED!"
            continue

    # Отображение информации на кадре
    status_text = f"Captured: {images_captured}/{TARGET_IMAGES}"
    if found_corners_on_frame:
        status_text += " | Corners Found: YES"
        text_color = (0, 255, 0) # Зеленый
    else:
        status_text += " | Corners Found: NO (Move board)"
        text_color = (0, 0, 255) # Красный

    cv2.putText(display_frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(display_frame, "Move board slowly. Press 'q' to quit.", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Camera Calibration - Frame Capture', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nСбор кадров прерван пользователем.")
        break

# --- Калибровка камеры (остальная часть кода без изменений) ---
if len(obj_points) >= MIN_IMAGES_FOR_CALIBRATION:
    print(f"\nСобрано {len(obj_points)} кадров. Выполняется калибровка...")
    image_size = gray.shape[::-1] # (width, height)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    if ret:
        calibration_successful = True
        print("\nКалибровка успешно завершена!")
        print(f"Средняя ошибка репроекции (RMS): {ret:.4f} пикселей")
        print("\nМатрица камеры (Intrinsic Matrix):")
        print(camera_matrix)
        print("\nКоэффициенты дисторсии:")
        print(dist_coeffs)

        calibration_data_file = '../../camera_calibration.npz'
        try:
            np.savez(calibration_data_file,
                     camera_matrix=camera_matrix,
                     dist_coeffs=dist_coeffs,
                     img_size=image_size,
                     reprojection_error=ret)
            print(f"\nПараметры калибровки сохранены в файл: {calibration_data_file}")
        except Exception as e:
            print(f"\nОшибка при сохранении файла {calibration_data_file}: {e}")
    else:
        print("\nОшибка: Функция calibrateCamera не смогла выполниться.")
else:
    print(f"\nНедостаточно кадров для калибровки (собрано {len(obj_points)}, минимум {MIN_IMAGES_FOR_CALIBRATION}).")
    print("Калибровка не будет выполнена.")

# --- Демонстрация устранения дисторсии ---
if calibration_successful:
    print("\n--- Демонстрация устранения дисторсии ---")
    print("Нажмите 'q', чтобы выйти из демонстрации.")

    h, w = actual_height, actual_width
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр для демонстрации.")
            break

        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Отображение результатов бок о бок
        if undistorted_frame.shape != frame.shape:
             frame_resized_for_compare = cv2.resize(frame, (undistorted_frame.shape[1], undistorted_frame.shape[0]))
             combined_view = np.hstack((frame_resized_for_compare, undistorted_frame))
        else:
             combined_view = np.hstack((frame, undistorted_frame))

        h_combined, w_combined = combined_view.shape[:2]
        w_original = frame.shape[1]

        cv2.putText(combined_view, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_view, "Undistorted", (w_original + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_view, "Press 'q' to quit", (20, h_combined - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        max_display_width = 1600
        if w_combined > max_display_width:
            scale = max_display_width / w_combined
            display_height = int(h_combined * scale)
            combined_view_resized = cv2.resize(combined_view, (max_display_width, display_height))
            cv2.imshow('Calibration Results (Original vs Undistorted)', combined_view_resized)
        else:
            cv2.imshow('Calibration Results (Original vs Undistorted)', combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Очистка ---
print("\nЗавершение работы...")
cap.release()
cv2.destroyAllWindows()
print("Программа завершена.")

