import asyncio
import websockets
import websockets.protocol
import ssl
import certifi
import json
import cv2
import numpy as np
import time
from datetime import datetime
from livekit import rtc  # Импорт LiveKit SDK

# --- Параметры калибровки камеры ---
camera_matrix = np.array(
    [
        [1.44464602e03, 0.00000000e00, 9.06452518e02],
        [0.00000000e00, 1.43108865e03, 4.68944978e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

dist_coeffs = np.array(
    [[-1.72642475e-01, -4.40297855e-01, 1.11232097e-04, 5.56096772e-04, 5.04451308e-01]]
)

TAG_SIZE = 0.04
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SEND_INTERVAL = 0.1

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

R_calib_matrix = None
t0_calib_vector = None
CALIBRATION_TAG_ID = 8

SERVER_URI = "wss://194.67.86.110:9001/apriltag"
AUTH_PAYLOAD = {
    "action": "apriltag_system_login",
    "payload": {"name": "dima_tag", "password": "jopapopa"},
}

car_tag_mapping = {}
layout_id_global = None
api_key_global = None
livekit_token_global = None
livekit_url = "wss://your-livekit-server.com"  # Замените на реальный URL LiveKit сервера

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


class CameraVideoSource(rtc.VideoSource):
    """Кастомный источник видео для LiveKit из камеры OpenCV"""

    def __init__(self, camera_index):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру с индексом {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    async def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Конвертируем BGR (OpenCV) в RGB (LiveKit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rtc.VideoFrame(
            width=frame_rgb.shape[1],
            height=frame_rgb.shape[0],
            data=frame_rgb.tobytes()
        )

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


async def start_livekit_stream(layout_id, token):
    """Подключение к LiveKit и начало трансляции видео"""
    room = rtc.Room()
    try:
        await room.connect(livekit_url, token)
        print(f"[{datetime.now()}] Успешно подключились к комнате LiveKit: {layout_id}")

        video_source = CameraVideoSource(CAMERA_INDEX)
        track = rtc.LocalVideoTrack.create_video_track("apriltag_camera", video_source)

        publication = await room.local_participant.publish_track(track)
        print(f"[{datetime.now()}] Видеопоток опубликован с ID: {publication.sid}")

        return room
    except Exception as e:
        print(f"[{datetime.now()}] Ошибка подключения к LiveKit: {e}")
        if room:
            await room.disconnect()
        return None


async def detect_and_send_apriltags(websocket):
    global R_calib_matrix, t0_calib_vector, layout_id_global, car_tag_mapping, api_key_global

    if not api_key_global or not layout_id_global:
        print(f"[{datetime.now()}] Ошибка: api_key или layout_id не инициализированы перед запуском детекции.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[{datetime.now()}] Ошибка: Не удалось открыть камеру с индексом {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(
        f"[{datetime.now()}] Камера {CAMERA_INDEX} инициализирована. Попытка установить разрешение {FRAME_WIDTH}x{FRAME_HEIGHT}")
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[{datetime.now()}] Фактическое разрешение камеры: {actual_width}x{actual_height}")

    print(f"[{datetime.now()}] Начинаю детекцию и отправку данных AprilTag для layout_id: {layout_id_global}...")

    last_send_time = time.monotonic()
    frame_count_for_fps = 0
    start_time_for_fps = time.monotonic()

    try:
        while websocket.state == websockets.protocol.State.OPEN:
            ret, frame = cap.read()
            if not ret:
                print(f"[{datetime.now()}] Ошибка: Не удалось получить кадр с камеры")
                await asyncio.sleep(0.1)
                continue

            current_time_mono = time.monotonic()
            frame_count_for_fps += 1
            corners, ids, _ = detector.detectMarkers(frame)
            tags_data_for_current_frame = []

            if ids is not None:
                for i, tag_id_array in enumerate(ids):
                    tag_id_raw = int(tag_id_array[0])
                    try:
                        marker_corners = corners[i][0]
                        obj_points = np.array(
                            [
                                [-TAG_SIZE / 2, TAG_SIZE / 2, 0],
                                [TAG_SIZE / 2, TAG_SIZE / 2, 0],
                                [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                                [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                            ],
                            dtype=np.float32,
                        )
                        ret_pnp, rvec, tvec = cv2.solvePnP(
                            obj_points,
                            marker_corners.astype(np.float32),
                            camera_matrix,
                            dist_coeffs,
                        )
                        if ret_pnp:
                            tvec_flat = tvec.flatten()
                            if tag_id_raw == CALIBRATION_TAG_ID:
                                if R_calib_matrix is None or t0_calib_vector is None:
                                    R_temp, _ = cv2.Rodrigues(rvec)
                                    R_calib_matrix = R_temp.copy()
                                    t0_calib_vector = tvec_flat.copy()
                                    print(
                                        f"[{datetime.now()}] Калибровочный тег ID {CALIBRATION_TAG_ID} обнаружен. R_calib_matrix и t0_calib_vector установлены.")
                                    print(
                                        f"t0_calib_vector (положение калибровочного тега в коорд. камеры): {t0_calib_vector}")
                                continue
                            elif R_calib_matrix is not None and t0_calib_vector is not None:
                                tvec_relative_to_camera = tvec_flat - t0_calib_vector
                                tvec_calib = R_calib_matrix.T @ tvec_relative_to_camera
                                car_id = car_tag_mapping.get(str(tag_id_raw))
                                if car_id:
                                    tags_data_for_current_frame.append(
                                        {
                                            "tag_id": str(tag_id_raw),
                                            "car_id": car_id,
                                            "x": float(tvec_calib[0]),
                                            "y": float(tvec_calib[1]),
                                            "z": float(tvec_calib[2]),
                                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                        }
                                    )
                    except Exception as e:
                        print(f"[{datetime.now()}] Ошибка обработки тега ID {tag_id_raw}: {e}")

            if current_time_mono - last_send_time >= SEND_INTERVAL:
                if websocket.state != websockets.protocol.State.OPEN:
                    print(f"[{datetime.now()}] WebSocket не открыт перед отправкой данных. Выход из цикла детекции.")
                    break

                elapsed_time_for_fps = current_time_mono - start_time_for_fps
                fps = frame_count_for_fps / elapsed_time_for_fps if elapsed_time_for_fps > 0 else 0
                message_to_send = {
                    "action": "april_tags_data",
                    "apikey": api_key_global,
                    "payload": {
                        "layout_id": layout_id_global,
                        "tags": tags_data_for_current_frame,
                        "fps": fps,
                    },
                }
                try:
                    await websocket.send(json.dumps(message_to_send))
                except websockets.exceptions.ConnectionClosed:
                    print(f"[{datetime.now()}] Соединение закрыто при отправке данных AprilTag.")
                    break
                except Exception as e:
                    print(f"[{datetime.now()}] Ошибка при отправке данных AprilTag: {e}")
                last_send_time = current_time_mono
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print(f"[{datetime.now()}] Соединение с сервером закрыто (в цикле детекции).")
    except asyncio.CancelledError:
        print(f"[{datetime.now()}] Цикл детекции отменен.")
    except Exception as e:
        print(f"[{datetime.now()}] Критическая ошибка в цикле детекции: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
        print(f"[{datetime.now()}] Камера освобождена и цикл детекции завершен.")


async def receive_updates(websocket):
    global car_tag_mapping
    try:
        while websocket.state == websockets.protocol.State.OPEN:
            try:
                message_str = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(message_str)
                action = data.get("action")
                if action == "car_status_updated":
                    payload = data.get("payload", {})
                    car_id = payload.get("car_id")
                    status = payload.get("status")
                    print(f"[{datetime.now()}] Обновление статуса машины {car_id}: {status}")
                elif action == "layout_config_updated":
                    payload = data.get("payload", {})
                    new_cars_config = payload.get("cars_config", [])
                    temp_mapping = {}
                    for car_conf in new_cars_config:
                        car_id_server = car_conf.get("car_id")
                        assigned_tag_id = car_conf.get("assigned_tag_id")
                        if car_id_server and assigned_tag_id is not None:
                            temp_mapping[str(assigned_tag_id)] = car_id_server
                    car_tag_mapping = temp_mapping
                    print(f"[{datetime.now()}] car_tag_mapping обновлен сервером: {car_tag_mapping}")
                else:
                    print(f"[{datetime.now()}] Получено сообщение: Action: {action}, Data: {str(data)[:200]}")
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                print(f"[{datetime.now()}] Получено не JSON сообщение в receive_updates: {message_str[:100]}")
            except websockets.exceptions.ConnectionClosed:
                print(f"[{datetime.now()}] Соединение закрыто (в receive_updates при попытке recv).")
                break
            except asyncio.CancelledError:
                print(f"[{datetime.now()}] receive_updates отменена.")
                raise
            except Exception as e:
                print(f"[{datetime.now()}] Ошибка в receive_updates: {e}")
                break
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        print(f"[{datetime.now()}] receive_updates отменена (внешний блок).")
    finally:
        print(f"[{datetime.now()}] Цикл приема обновлений завершен.")


async def main_client():
    global layout_id_global, car_tag_mapping, api_key_global, livekit_token_global

    while True:
        active_websocket = None
        livekit_room = None
        tasks = []
        try:
            async with websockets.connect(
                    SERVER_URI, ssl=ssl_context, ping_interval=20, ping_timeout=10
            ) as websocket:
                active_websocket = websocket
                print(f"[{datetime.now()}] Подключен к серверу: {SERVER_URI}")
                await websocket.send(json.dumps(AUTH_PAYLOAD))
                print(f"[{datetime.now()}] Запрос авторизации отправлен.")
                auth_response_str = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                auth_response = json.loads(auth_response_str)
                print(f"[{datetime.now()}] Ответ авторизации получен: {auth_response}")

                if auth_response.get("status") != "success":
                    print(
                        f"[{datetime.now()}] Ошибка авторизации: {auth_response.get('message', 'Неизвестная ошибка')}")
                    raise ConnectionAbortedError("Auth failed, will retry")

                data = auth_response.get("data", {})
                api_key_global = data.get("api_key")
                layout_id_global = data.get("layout_id")
                livekit_token_global = data.get("livekit_token")  # Новое поле
                cars_config = data.get("cars_config", [])

                if not api_key_global or not layout_id_global or not livekit_token_global:
                    print(f"[{datetime.now()}] Не получены необходимые данные в ответе авторизации.")
                    raise ConnectionAbortedError("Invalid auth response, will retry")

                # Инициализация LiveKit соединения
                livekit_room = await start_livekit_stream(layout_id_global, livekit_token_global)
                if not livekit_room:
                    print(f"[{datetime.now()}] Не удалось инициализировать соединение с LiveKit")
                    raise ConnectionAbortedError("LiveKit connection failed")

                car_tag_mapping.clear()
                for car_conf in cars_config:
                    car_id = car_conf.get("car_id")
                    assigned_tag_id_val = car_conf.get("assigned_tag_id")
                    if car_id and assigned_tag_id_val is not None:
                        car_tag_mapping[str(assigned_tag_id_val)] = car_id

                print(f"[{datetime.now()}] car_tag_mapping сформирован: {car_tag_mapping}")
                print(
                    f"[{datetime.now()}] Авторизация успешна. API Key: ...{api_key_global[-5:]}, Layout ID: {layout_id_global}")

                receive_task = asyncio.create_task(receive_updates(websocket), name="ReceiveUpdatesTask")
                detect_task = asyncio.create_task(detect_and_send_apriltags(websocket), name="DetectTask")
                tasks = [receive_task, detect_task]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                print(f"[{datetime.now()}] Одна из основных задач ({[t.get_name() for t in done]}) завершилась.")
                for task in pending:
                    if not task.done():
                        print(f"[{datetime.now()}] Отмена ожидающей задачи: {task.get_name()}")
                        task.cancel()
                if pending:
                    results_pending = await asyncio.gather(*pending, return_exceptions=True)
                    for i, res_p in enumerate(results_pending):
                        if isinstance(res_p, Exception) and not isinstance(res_p, asyncio.CancelledError):
                            print(
                                f"[{datetime.now()}] Исключение в ожидающей задаче {pending[i].get_name()}: {type(res_p).__name__} - {res_p}")

                for task_done in done:
                    try:
                        task_done.result()
                    except asyncio.CancelledError:
                        print(f"[{datetime.now()}] Завершённая задача {task_done.get_name()} была отменена.")
                    except websockets.exceptions.ConnectionClosed:
                        print(f"[{datetime.now()}] Завершённая задача {task_done.get_name()} из-за ConnectionClosed.")
                    except Exception as e_done_task:
                        print(
                            f"[{datetime.now()}] Ошибка в завершённой задаче {task_done.get_name()}: {type(e_done_task).__name__} - {e_done_task}")

        except websockets.exceptions.ConnectionClosedOK as e:
            print(f"[{datetime.now()}] Соединение закрыто (OK): {e}")
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosed) as e:
            print(f"[{datetime.now()}] Соединение закрыто (ошибка): {type(e).__name__} - {e}")
        except ConnectionRefusedError:
            print(f"[{datetime.now()}] Отказ в соединении с {SERVER_URI}. Сервер недоступен?")
        except asyncio.TimeoutError:
            print(f"[{datetime.now()}] Таймаут при подключении или авторизации.")
        except json.JSONDecodeError as e:
            print(f"[{datetime.now()}] Ошибка декодирования JSON ответа сервера: {e}")
        except ssl.SSLError as e:
            print(f"[{datetime.now()}] Ошибка SSL: {e}. Проверьте сертификат сервера или настройки ssl_context.")
        except ConnectionAbortedError as e:
            print(f"[{datetime.now()}] Логическое прерывание соединения: {e}")
        except Exception as e:
            print(f"[{datetime.now()}] Непредвиденная ошибка в main_client: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[{datetime.now()}] Сессия WebSocket завершена.")

            # Отключение от LiveKit
            if livekit_room:
                print(f"[{datetime.now()}] Отключение от комнаты LiveKit...")
                await livekit_room.disconnect()

            # Отмена задач
            for task_to_cancel in tasks:
                if task_to_cancel and not task_to_cancel.done():
                    print(f"[{datetime.now()}] Отмена задачи {task_to_cancel.get_name()} в finally.")
                    task_to_cancel.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            print(f"[{datetime.now()}] Попытка переподключения через 5 секунд...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main_client())
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] Клиент остановлен пользователем (KeyboardInterrupt).")
    finally:
        print(f"[{datetime.now()}] Завершение работы скрипта AprilTag клиента.")