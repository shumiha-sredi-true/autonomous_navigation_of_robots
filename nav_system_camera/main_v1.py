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
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from av import VideoFrame

# --- Параметры калибровки камеры ---
camera_matrix = np.array([
    [1.44464602e03, 0.00000000e00, 9.06452518e02],
    [0.00000000e00, 1.43108865e03, 4.68944978e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00]
])

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

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


class CameraVideoStreamTrack(VideoStreamTrack):
    """Кастомный VideoStreamTrack для передачи видео с камеры через WebRTC"""

    def __init__(self, camera_index=CAMERA_INDEX):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open tech_v with index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"Camera initialized with resolution {FRAME_WIDTH}x{FRAME_HEIGHT}")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from tech_v")

        # Конвертируем кадр из BGR в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class WebRTCClient:
    def __init__(self):
        self.pc = RTCPeerConnection()
        self.video_track = None
        self.webrtc_ready = False

    async def setup(self, websocket, api_key, layout_id):
        """Настройка WebRTC соединения"""
        self.video_track = CameraVideoStreamTrack()
        self.pc.addTrack(self.video_track)

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                candidate_msg = {
                    "action": "webrtc_ice",
                    "apikey": api_key,
                    "payload": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex
                    }
                }
                await websocket.send(json.dumps(candidate_msg))

        # Создаем offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Отправляем offer на сервер
        webrtc_offer = {
            "action": "webrtc_offer",
            "apikey": api_key,
            "payload": {
                "layout_id": layout_id,
                "sdp": self.pc.localDescription.sdp,
                "type": "offer"
            }
        }

        await websocket.send(json.dumps(webrtc_offer))
        self.webrtc_ready = True

    async def handle_answer(self, answer_sdp):
        """Обработка ответа от сервера"""
        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self.pc.setRemoteDescription(answer)

    async def add_ice_candidate(self, candidate_data):
        """Добавление ICE кандидата"""
        candidate = RTCIceCandidate(
            candidate_data["candidate"],
            candidate_data["sdpMid"],
            candidate_data["sdpMLineIndex"]
        )
        await self.pc.addIceCandidate(candidate)

    async def close(self):
        """Закрытие соединения"""
        await self.pc.close()
        self.webrtc_ready = False


async def detect_and_send_apriltags(websocket, webrtc_client):
    global R_calib_matrix, t0_calib_vector, layout_id_global, car_tag_mapping, api_key_global

    if not api_key_global or not layout_id_global:
        print(f"[{datetime.now()}] Ошибка: api_key или layout_id не инициализированы")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[{datetime.now()}] Ошибка: Не удалось открыть камеру {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"[{datetime.now()}] Камера {CAMERA_INDEX} инициализирована. Разрешение: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[{datetime.now()}] Начинаю детекцию AprilTag для layout_id: {layout_id_global}...")

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
                        obj_points = np.array([
                            [-TAG_SIZE / 2, TAG_SIZE / 2, 0],
                            [TAG_SIZE / 2, TAG_SIZE / 2, 0],
                            [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                            [-TAG_SIZE / 2, -TAG_SIZE / 2, 0]
                        ], dtype=np.float32)

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
                                    print(f"[{datetime.now()}] Калибровочный тег ID {CALIBRATION_TAG_ID} обнаружен")
                                continue
                            elif (R_calib_matrix is not None and t0_calib_vector is not None):
                                tvec_relative_to_camera = tvec_flat - t0_calib_vector
                                tvec_calib = R_calib_matrix.T @ tvec_relative_to_camera
                                car_id = car_tag_mapping.get(str(tag_id_raw))
                                if car_id:
                                    tags_data_for_current_frame.append({
                                        "tag_id": str(tag_id_raw),
                                        "car_id": car_id,
                                        "x": float(tvec_calib[0]),
                                        "y": float(tvec_calib[1]),
                                        "z": float(tvec_calib[2]),
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    })
                    except Exception as e:
                        print(f"[{datetime.now()}] Ошибка обработки тега ID {tag_id_raw}: {e}")

            if current_time_mono - last_send_time >= SEND_INTERVAL:
                if websocket.state != websockets.protocol.State.OPEN:
                    print(f"[{datetime.now()}] WebSocket не открыт. Выход из цикла детекции.")
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
                    print(f"[{datetime.now()}] Соединение закрыто при отправке данных.")
                    break
                except Exception as e:
                    print(f"[{datetime.now()}] Ошибка при отправке данных: {e}")

                last_send_time = current_time_mono
                frame_count_for_fps = 0
                start_time_for_fps = current_time_mono

            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"[{datetime.now()}] Ошибка в цикле детекции: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        print(f"[{datetime.now()}] Камера освобождена.")


async def handle_webrtc_signaling(websocket, webrtc_client):
    """Обработка WebRTC сигналов"""
    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get("action")

            if action == "webrtc_answer" and webrtc_client:
                await webrtc_client.handle_answer(data["payload"]["sdp"])

            elif action == "webrtc_ice" and webrtc_client:
                await webrtc_client.add_ice_candidate(data["payload"])

    except Exception as e:
        print(f"[{datetime.now()}] Ошибка обработки WebRTC сигналов: {e}")


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
                    print(
                        f"[{datetime.now()}] Обновление статуса машины {payload.get('car_id')}: {payload.get('status')}")

                elif action == "layout_config_updated":
                    payload = data.get("payload", {})
                    new_cars_config = payload.get("cars_config", [])
                    car_tag_mapping = {
                        str(car_conf["assigned_tag_id"]): car_conf["car_id"]
                        for car_conf in new_cars_config
                        if car_conf.get("assigned_tag_id") is not None
                    }
                    print(f"[{datetime.now()}] car_tag_mapping обновлен: {car_tag_mapping}")

                else:
                    print(f"[{datetime.now()}] Получено сообщение: {action}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[{datetime.now()}] Ошибка в receive_updates: {e}")
                break

    except Exception as e:
        print(f"[{datetime.now()}] Ошибка в цикле receive_updates: {e}")


async def main_client():
    global layout_id_global, car_tag_mapping, api_key_global

    while True:
        webrtc_client = None
        try:
            async with websockets.connect(
                    SERVER_URI,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10
            ) as websocket:
                print(f"[{datetime.now()}] Подключен к серверу: {SERVER_URI}")

                # Авторизация
                await websocket.send(json.dumps(AUTH_PAYLOAD))
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)

                if auth_data.get("status") != "success":
                    print(f"[{datetime.now()}] Ошибка авторизации")
                    raise ConnectionAbortedError("Auth failed")

                api_key_global = auth_data["data"].get("api_key")
                layout_id_global = auth_data["data"].get("layout_id")
                cars_config = auth_data["data"].get("cars_config", [])

                if not api_key_global or not layout_id_global:
                    print(f"[{datetime.now()}] Не получен api_key или layout_id")
                    raise ConnectionAbortedError("Invalid auth response")

                car_tag_mapping = {
                    str(car_conf["assigned_tag_id"]): car_conf["car_id"]
                    for car_conf in cars_config
                    if car_conf.get("assigned_tag_id") is not None
                }

                print(f"[{datetime.now()}] Авторизация успешна. Layout ID: {layout_id_global}")

                # Инициализация WebRTC
                webrtc_client = WebRTCClient()
                await webrtc_client.setup(websocket, api_key_global, layout_id_global)

                # Запуск задач
                detect_task = asyncio.create_task(detect_and_send_apriltags(websocket, webrtc_client))
                receive_task = asyncio.create_task(receive_updates(websocket))
                signaling_task = asyncio.create_task(handle_webrtc_signaling(websocket, webrtc_client))

                await asyncio.wait(
                    [detect_task, receive_task, signaling_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Очистка
                for task in [detect_task, receive_task, signaling_task]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except:
                            pass

                if webrtc_client:
                    await webrtc_client.close()

        except (websockets.exceptions.ConnectionClosed, ConnectionAbortedError) as e:
            print(f"[{datetime.now()}] Соединение закрыто: {e}")
        except Exception as e:
            print(f"[{datetime.now()}] Ошибка в main_client: {e}")
        finally:
            print(f"[{datetime.now()}] Попытка переподключения через 5 секунд...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main_client())
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] Клиент остановлен пользователем")
    finally:
        print(f"[{datetime.now()}] Завершение работы скрипта")