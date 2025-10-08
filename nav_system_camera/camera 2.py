import asyncio
import json
import logging
import time
import ssl
import certifi
import traceback

import cv2
import numpy as np

import websockets
from livekit import rtc

# --- Конфигурация ---
BACKEND_WEBSOCKET_URL = "wss://cloudbots.hackmpei.ru:9001/robot"

ROBOT_NAME = "car4"
ROBOT_PASSWORD = "jopapopa"

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS = 30

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class RobotClient:
    """
    Класс клиента-робота, логинится на бекенд, подключается к LiveKit и публикует видео с камеры.
    """
    def __init__(self, url, name, password):
        self.backend_url = url
        self.name = name
        self.password = password
        self.ws_connection = None
        self.livekit_url = None
        self.livekit_token = None
        self.room = None
        self.stop_event = asyncio.Event()

        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def _login(self) -> bool:
        """Логинится на бекенде и получает livekit_url + livekit_token"""
        logging.info(f"Попытка входа для робота: {self.name}")
        login_payload = {
            "action": "device_login",
            "payload": {"name": self.name, "password": self.password},
        }

        try:
            await self.ws_connection.send(json.dumps(login_payload))
        except Exception as e:
            logging.error(f"Ошибка при отправке логина: {e}")
            return False

        try:
            response_raw = await asyncio.wait_for(self.ws_connection.recv(), timeout=10.0)
            response = json.loads(response_raw)
            if response.get("action") == "device_login" and response.get("status") == "success":
                data = response.get("data", {})
                self.livekit_url = data.get("livekit_url")
                self.livekit_token = data.get("livekit_token")
                if self.livekit_url and self.livekit_token:
                    logging.info("Успешный вход! Получены данные для LiveKit.")
                    return True
        except Exception as e:
            logging.error(f"Ошибка при логине: {e}")
            traceback.print_exc()
        return False

    async def _publish_camera_video(self):
        """Захватывает видео с камеры и публикует в LiveKit"""
        cap = cv2.VideoCapture(0)  # 0 = встроенная камера
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        if not cap.isOpened():
            logging.error("Не удалось открыть камеру!")
            return

        video_source = rtc.VideoSource(VIDEO_WIDTH, VIDEO_HEIGHT)
        track = rtc.LocalVideoTrack.create_video_track("tech_v", video_source)

        try:
            await self.room.local_participant.publish_track(track)
            logging.info("Видеопоток с камеры успешно опубликован в LiveKit.")
        except Exception as e:
            logging.error(f"Ошибка при публикации трека: {e}")
            traceback.print_exc()
            return

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                # OpenCV даёт BGR → переводим в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buf = frame_rgb.tobytes()

                video_frame = rtc.VideoFrame(
                    VIDEO_WIDTH,
                    VIDEO_HEIGHT,
                    rtc.VideoBufferType.RGB24,
                    buf
                )
                video_source.capture_frame(video_frame, timestamp_us=int(time.time() * 1_000_000))

                await asyncio.sleep(1 / FPS)
        finally:
            cap.release()
            logging.info("Камера освобождена.")

    async def _listen_to_backend(self):
        try:
            async for message in self.ws_connection:
                logging.info(f"Сообщение от бэкенда: {message}")
        except websockets.exceptions.ConnectionClosed:
            logging.info("Соединение с бэкендом закрыто.")
        finally:
            self.stop_event.set()

    async def run(self):
        ssl_arg = self.ssl_context if self.backend_url.lower().startswith("wss://") else None
        try:
            async with websockets.connect(self.backend_url, ssl=ssl_arg) as ws:
                self.ws_connection = ws
                if not await self._login():
                    return

                self.room = rtc.Room()
                logging.info(f"Подключение к LiveKit: {self.livekit_url}")
                await self.room.connect(self.livekit_url, self.livekit_token)
                logging.info("Успешно подключен к комнате LiveKit.")

                video_task = asyncio.create_task(self._publish_camera_video())
                listen_task = asyncio.create_task(self._listen_to_backend())
                await asyncio.gather(video_task, listen_task)

        except Exception as e:
            logging.error(f"Ошибка в run: {e}")
            traceback.print_exc()
        finally:
            try:
                if self.room and self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    await self.room.disconnect()
            except Exception:
                pass


async def main():
    client = RobotClient(BACKEND_WEBSOCKET_URL, ROBOT_NAME, ROBOT_PASSWORD)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Программа остановлена пользователем.")
