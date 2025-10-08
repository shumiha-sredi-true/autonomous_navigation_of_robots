import asyncio
import websockets
import websockets.exceptions
import ssl
import certifi
import json
import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

import numpy as np
import time
import traceback
from datetime import datetime
from livekit import rtc


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
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SEND_INTERVAL = 0.1  # как часто шлём теги (сек)

# детектор AprilTag
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

R_calib_matrix = None
t0_calib_vector = None
CALIBRATION_TAG_ID = 13

SERVER_URI = "wss://194.67.86.110:9001/apriltag"
# AUTH_PAYLOAD = {
#     "action": "apriltag_system_login",
#     "payload": {"name": "dima_tag", "password": "jopapopa"},
# }
AUTH_PAYLOAD = {
    "action": "apriltag_system_login",
    # "payload": {"name": "dima_tag", "password": "jopapopa"},
    "payload": {"name": "test", "password": "test"},
}


car_tag_mapping = {}
layout_id_global = None
api_key_global = None
livekit_token_global = None
LIVEKIT_URL = "ws://194.67.86.110:7880"

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Общие объекты для обмена кадром
latest_frame = None
frame_lock = None  # asyncio.Lock, инициализируется в main_client()

# --- Храним предыдущие позиции для вычисления скорости ---
last_positions = {}  # {car_id: {"pos": np.array([x,y,z]), "time": t, "vel": np.array([vx,vy,vz])}}


async def detect_and_send_apriltags(websocket):
    """Читает latest_frame (обновлённый publish_video_to_livekit), находит теги, вычисляет скорость и шлёт данные."""
    global R_calib_matrix, t0_calib_vector, layout_id_global, car_tag_mapping, api_key_global
    global latest_frame, frame_lock, last_positions

    print(f"[{datetime.now()}] -> START detect_and_send_apriltags")
    if not api_key_global or not layout_id_global:
        print(f"[{datetime.now()}] Ошибка: api_key или layout_id не инициализированы.")
        return

    last_send_time = time.monotonic()

    try:
        while True:
            if getattr(websocket, "closed", False):
                code = getattr(websocket, "close_code", None)
                reason = getattr(websocket, "close_reason", None)
                print(f"[{datetime.now()}] WebSocket закрыт (detect). close_code={code}, reason={reason}")
                break

            # Берём последнюю доступную копию кадра
            async with frame_lock:
                frame_local = None if latest_frame is None else latest_frame.copy()

            if frame_local is None:
                await asyncio.sleep(0.02)
                continue

            current_time_mono = time.monotonic()
            corners, ids, _ = detector.detectMarkers(frame_local)
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

                        success, rvec, tvec = cv2.solvePnP(
                            obj_points.astype(np.float32),
                            marker_corners.astype(np.float32),
                            camera_matrix,
                            dist_coeffs,
                        )
                        if not success:
                            continue

                        tvec_flat = tvec.flatten()
                        if tag_id_raw == CALIBRATION_TAG_ID:
                            if R_calib_matrix is None or t0_calib_vector is None:
                                R_temp, _ = cv2.Rodrigues(rvec)
                                R_calib_matrix = R_temp.copy()
                                t0_calib_vector = tvec_flat.copy()
                                print(f"[{datetime.now()}] Калибровочный тег найден, матрица сохранена.")
                            continue

                        if R_calib_matrix is not None and t0_calib_vector is not None:
                            tvec_relative_to_camera = tvec_flat - t0_calib_vector
                            tvec_calib = R_calib_matrix.T @ tvec_relative_to_camera*1.3
                            car_id = car_tag_mapping.get(str(tag_id_raw))
                            if car_id:
                                now = time.time()
                                velocity = np.array([0.0, 0.0, 0.0])
                                if car_id in last_positions:
                                    prev = last_positions[car_id]
                                    dt = now - prev["time"]
                                    if dt > 0:
                                        velocity = (tvec_calib - prev["pos"]) / dt
                                last_positions[car_id] = {"pos": tvec_calib.copy(), "time": now, "vel": velocity.copy()}

                                tags_data_for_current_frame.append(
                                    {
                                        "tag_id": str(tag_id_raw),
                                        "car_id": car_id,
                                        "x": float(tvec_calib[0]),
                                        "y": float(tvec_calib[1]),
                                        "z": float(tvec_calib[2]),
                                        "vx": float(velocity[0]),
                                        "vy": float(velocity[1]),
                                        "vz": float(velocity[2]),
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    }
                                )
                    except Exception as e:
                        print(f"[{datetime.now()}] Ошибка обработки тега {tag_id_raw}: {e}")
                        traceback.print_exc()

            if current_time_mono - last_send_time >= SEND_INTERVAL and tags_data_for_current_frame:
                message_to_send = {
                    "action": "april_tags_data",
                    "apikey": api_key_global,
                    "payload": {"layout_id": layout_id_global, "tags": tags_data_for_current_frame},
                }
                print(message_to_send)
                try:
                    await websocket.send(json.dumps(message_to_send))
                    last_send_time = current_time_mono
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[{datetime.now()}] Ошибка отправки: WebSocket закрылся при send: code={getattr(e,'code',None)}, reason={getattr(e,'reason',None)}")
                    break
                except Exception as e:
                    print(f"[{datetime.now()}] Неизвестная ошибка отправки: {e}")
                    traceback.print_exc()
                    break

            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        print(f"[{datetime.now()}] detect_and_send_apriltags cancelled.")
    except Exception as e:
        print(f"[{datetime.now()}] Error in detect_and_send_apriltags: {e}")
        traceback.print_exc()
    finally:
        print(f"[{datetime.now()}] <- END detect_and_send_apriltags")


async def receive_updates(websocket):
    global car_tag_mapping
    print(f"[{datetime.now()}] -> START receive_updates")
    try:
        async for message_str in websocket:
            data = json.loads(message_str)
            action = data.get("action")
            print(2)
            print(data)
            if action == "layout_config_updated":
                payload = data.get("payload", {})
                temp_mapping = {}
                for car_conf in payload.get("cars_config", []):
                    car_id_server = car_conf.get("car_id")
                    assigned_tag_id = car_conf.get("assigned_tag_id")
                    if car_id_server and assigned_tag_id is not None:
                        temp_mapping[str(assigned_tag_id)] = car_id_server
                car_tag_mapping = temp_mapping
                print(f"[{datetime.now()}] mapping updated: {car_tag_mapping}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[{datetime.now()}] receive_updates: WebSocket closed: code={getattr(e,'code',None)}, reason={getattr(e,'reason',None)}")
    except asyncio.CancelledError:
        print(f"[{datetime.now()}] receive_updates cancelled.")
    except Exception as e:
        print(f"[{datetime.now()}] Error in receive_updates: {e}")
        traceback.print_exc()
    finally:
        print(f"[{datetime.now()}] <- END receive_updates")


async def publish_video_to_livekit(cap):
    """Читает кадры, рисует AprilTag и отправляет в LiveKit."""
    global layout_id_global, livekit_token_global, LIVEKIT_URL, latest_frame, frame_lock, last_positions

    print(f"[{datetime.now()}] -> START publish_video_to_livekit")
    room = rtc.Room()
    video_source = None

    try:
        if not livekit_token_global or not layout_id_global:
            print(f"[{datetime.now()}] Нет livekit_token или layout_id — видео не будет публиковаться.")
            return

        print(f"[{datetime.now()}] Попытка подключения к LiveKit {LIVEKIT_URL} ...")
        await room.connect(LIVEKIT_URL, livekit_token_global)
        print(f"[{datetime.now()}] УСПЕХ: подключены к LiveKit.")

        video_source = rtc.VideoSource(FRAME_WIDTH, FRAME_HEIGHT)
        track = rtc.LocalVideoTrack.create_video_track("tech_v", video_source)
        await room.local_participant.publish_track(track)
        print(f"[{datetime.now()}] Track published; waiting 0.5s for negotiation...")
        await asyncio.sleep(0.5)

        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.02)
                continue

            # Копия для отрисовки
            draw_frame = frame.copy()

            # Поиск AprilTag прямо здесь для отображения
            corners, ids, _ = detector.detectMarkers(draw_frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(draw_frame, corners, ids)
                for i, tag_id_array in enumerate(ids):
                    tag_id_raw = int(tag_id_array[0])
                    car_id = car_tag_mapping.get(str(tag_id_raw))
                    text = f"Tag {tag_id_raw}"
                    if car_id and car_id in last_positions:
                        v = last_positions[car_id]["vel"]
                        text += f" v=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})"
                    corner = corners[i][0][0]  # левый верхний угол
                    cv2.putText(draw_frame, text, (int(corner[0]), int(corner[1]-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Обновляем latest_frame для детектора
            async with frame_lock:
                latest_frame = frame.copy()

            try:
                frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[{datetime.now()}] Ошибка cvtColor: {e}")
                await asyncio.sleep(0.01)
                continue

            vf = rtc.VideoFrame(FRAME_WIDTH, FRAME_HEIGHT, rtc.VideoBufferType.RGB24, frame_rgb.tobytes())
            ts = int(time.time() * 1_000_000)
            try:
                video_source.capture_frame(vf, timestamp_us=ts)
            except Exception as e:
                print(f"[{datetime.now()}] capture_frame error: {e}")
                traceback.print_exc()
                break

            await asyncio.sleep(1/30)  # ~30 FPS

    except asyncio.CancelledError:
        print(f"[{datetime.now()}] publish_video_to_livekit cancelled.")
    except Exception as e:
        print(f"[{datetime.now()}] CRITICAL error in publish_video_to_livekit: {e}")
        traceback.print_exc()
    finally:
        try:
            if video_source is not None and hasattr(video_source, "aclose"):
                await video_source.aclose()
        except Exception:
            pass
        try:
            await room.disconnect()
        except Exception:
            pass
        print(f"[{datetime.now()}] <- END publish_video_to_livekit")


async def main_client():
    global layout_id_global, car_tag_mapping, api_key_global, livekit_token_global, LIVEKIT_URL
    global frame_lock, latest_frame

    backoff = 5
    max_backoff = 60

    while True:
        receive_task = None
        detect_task = None
        livekit_task = None
        cap = None
        frame_lock = asyncio.Lock()
        latest_frame = None

        try:
            print(f"[{datetime.now()}] Connecting to {SERVER_URI} ...")
            async with websockets.connect(SERVER_URI, ssl=ssl_context, ping_interval=20, ping_timeout=10) as websocket:
                print(f"[{datetime.now()}] Connected; ws.closed={getattr(websocket,'closed',None)}")
                await websocket.send(json.dumps(AUTH_PAYLOAD))
                auth_response_str = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                auth_response = json.loads(auth_response_str)
                print(f"[{datetime.now()}] auth response: {auth_response}")

                if auth_response.get("status") != "success":
                    raise ConnectionAbortedError("Auth failed")

                data = auth_response.get("data", {})
                api_key_global = data.get("api_key")
                layout_id_global = data.get("layout_id")
                livekit_token_global = data.get("livekit_token")
                livekit_url_from_server = data.get("livekit_url")
                if livekit_url_from_server:
                    LIVEKIT_URL = livekit_url_from_server

                cars_config = data.get("cars_config", [])
                car_tag_mapping.clear()
                for car_conf in cars_config:
                    car_id = car_conf.get("car_id")
                    assigned_tag_id_val = car_conf.get("assigned_tag_id")
                    if car_id and assigned_tag_id_val is not None:
                        car_tag_mapping[str(assigned_tag_id_val)] = car_id

                print(f"[{datetime.now()}] Auth OK. LIVEKIT_URL={LIVEKIT_URL}")

                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                if not cap.isOpened():
                    print(f"[{datetime.now()}] ERROR: cannot open tech_v {CAMERA_INDEX}. Video/detection disabled.")
                    receive_task = asyncio.create_task(receive_updates(websocket))
                    done, pending = await asyncio.wait([receive_task], return_when=asyncio.FIRST_COMPLETED)
                else:
                    receive_task = asyncio.create_task(receive_updates(websocket))
                    livekit_task = asyncio.create_task(publish_video_to_livekit(cap))
                    detect_task = asyncio.create_task(detect_and_send_apriltags(websocket))

                    done, pending = await asyncio.wait(
                        [receive_task, livekit_task, detect_task], return_when=asyncio.FIRST_COMPLETED
                    )

                print(f"[{datetime.now()}] One task finished, cancelling others...")
                for t in [receive_task, livekit_task, detect_task]:
                    if t and not t.done():
                        t.cancel()

                await asyncio.gather(*(t for t in [receive_task, livekit_task, detect_task] if t), return_exceptions=True)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[{datetime.now()}] main websocket closed: code={getattr(e,'code',None)}, reason={getattr(e,'reason',None)}")
            traceback.print_exc()
        except ConnectionAbortedError as e:
            print(f"[{datetime.now()}] Connection aborted: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"[{datetime.now()}] Unexpected error in main_client: {e}")
            traceback.print_exc()
        finally:
            try:
                if cap is not None and cap.isOpened():
                    cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

            print(f"[{datetime.now()}] Reconnect after {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)


if __name__ == "__main__":
    try:
        asyncio.run(main_client())
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] stopped by user")
    except Exception:
        traceback.print_exc()
    finally:
        print("exiting")
