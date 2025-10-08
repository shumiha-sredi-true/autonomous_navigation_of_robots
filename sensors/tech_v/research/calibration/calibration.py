#!/usr/bin/env python3
"""
camera_calibration.py

Краткое: Калибровка камеры по множеству фотографий шахматной доски.

Как использовать:
    python camera_calibration.py

Параметры:
  (без параметров) — скрипт автоматически возьмёт все изображения из текущей директории.

Используется шахматная доска:
  - количество внутренних углов по ширине: 7
  - количество внутренних углов по высоте: 9
  - размер клетки: 20 мм (0.020 м)

Выход:
  - camera_matrix (3x3)
  - dist_coeffs (k1,k2,p1,p2,k3...)
  - rvecs, tvecs для каждой картинки
  - средняя ошибка репроекции

Требования:
  pip install opencv-python numpy

Автор: автогенерация
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np

# фиксированные параметры шахматной доски
BOARD_W = 7
BOARD_H = 9
SQUARE_SIZE = 0.020  # метры (20 мм)

# поддерживаемые расширения файлов
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def collect_image_paths():
    cwd = Path(os.getcwd())
    paths = []
    for ext in IMG_EXTS:
        paths.extend(sorted(cwd.glob(ext)))
    if not paths:
        raise FileNotFoundError(f"No images found in {cwd}")
    return [str(p) for p in paths]


def prepare_object_points(board_w, board_h, square_size):
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate(images, board_w, board_h, square_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    pattern_size = (board_w, board_h)

    objp = prepare_object_points(board_w, board_h, square_size)

    objpoints = []
    imgpoints = []
    img_shape = None
    used_images = []

    for fname in images:
        img = imread_unicode(fname)
        if img is None:
            print(f"Warning: couldn't read {fname}, skipping")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            try:
                found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
            except Exception:
                pass

        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)
            objpoints.append(objp)
            imgpoints.append(corners_sub)
            used_images.append(fname)

            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_sub, True)
            cv2.imshow('corners', vis)
            key = cv2.waitKey(300) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                break
        else:
            print(f"Chessboard not found in {fname}")

    cv2.destroyAllWindows()

    if len(objpoints) < 3:
        raise RuntimeError(f"Need at least 3 successful detections for reliable calibration, got {len(objpoints)}")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    result = {
        'ret': float(ret),
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'mean_error': float(mean_error),
        'used_images': used_images,
        'image_size': img_shape,
    }

    return result


def save_result(result, out_path):
    out_path = Path(out_path)
    if out_path.suffix == '.npz':
        np.savez(
            str(out_path),
            camera_matrix=result['camera_matrix'],
            dist_coeffs=result['dist_coeffs'],
            rvecs=result['rvecs'],
            tvecs=result['tvecs'],
            mean_error=result['mean_error']
        )
        print(f"Saved npz to {out_path}")
    else:
        fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
        fs.write('camera_matrix', result['camera_matrix'])
        fs.write('dist_coeffs', result['dist_coeffs'])
        fs.write('mean_error', result['mean_error'])
        fs.write('image_width', int(result['image_size'][0]))
        fs.write('image_height', int(result['image_size'][1]))
        fs.startWriteStruct('used_images', cv2.FILE_NODE_SEQ)
        for p in result.get('used_images', []):
            fs.write('', p)
        fs.endWriteStruct()
        fs.release()
        print(f"Saved YAML to {out_path}")


def undistort_and_show(image_path, camera_matrix, dist_coeffs):
    img = imread_unicode(image_path)
    if img is None:
        print(f"Can't read {image_path}")
        return
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    if w > 0 and h > 0:
        undistorted = undistorted[y:y+h, x:x+w]
    combined = np.hstack([cv2.resize(img, (undistorted.shape[1], undistorted.shape[0])), undistorted])
    cv2.imshow('orig | undistorted', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    images = collect_image_paths()

    print(f"Found {len(images)} images in current directory. Using board {BOARD_W}x{BOARD_H}, square size {SQUARE_SIZE} m")

    result = calibrate(images, BOARD_W, BOARD_H, SQUARE_SIZE)

    print("Calibration done")
    print(f"RMS re-projection error: {result['mean_error']}")
    print("Camera matrix:\n", result['camera_matrix'])
    print("Distortion coefficients:\n", result['dist_coeffs'].ravel())

    save_result(result, "calib_result.yml")

    if result['used_images']:
        undistort_and_show(result['used_images'][0], result['camera_matrix'], result['dist_coeffs'])


if __name__ == '__main__':
    main()