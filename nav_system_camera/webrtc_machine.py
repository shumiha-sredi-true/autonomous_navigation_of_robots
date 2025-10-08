import asyncio
import base64
import json
import psutil
import ssl
import os
import subprocess
import time
from threading import Thread
from time import sleep
import cv2
import numpy as np
import RPi.GPIO as GPIO
import serial
import websocket
import websockets
import mmap
import tempfile
import atexit
import logging
import av  # для формирования VideoFrame

# Импорт библиотеки для WebRTC
from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCIceCandidate,
    RTCIceServer,
)

# Импорт пользовательских модулей
from kalman_filter.log_reader import reader_logs
from kalman_filter.Kalman_filter import EKF3

# ==============================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКИ
# ==============================
global_frame = None          # Глобальный кадр для других целей
ws = None                    # Глобальное WebSocket-соединение
webrtc_client = None         # Глобальный экземпляр WebRTCClient

# Параметры кадра
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CHANNELS = 3
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS

# Временная директория и файлы для двойного буфера
temp_dir = tempfile.gettempdir()
SHM_FILENAMES = [os.path.join(temp_dir, "shm_frame0"), os.path.join(temp_dir, "shm_frame1")]

# ------------------------------
# ПАРАМЕТРЫ РОБОТА
# ------------------------------
login = "robot2"
password = "jopapopa"
api = ""
car_id = ""
user_id = ""
All_cars = []
status_connection = False
stop = 1
uwb_stat = True
mainloop = True

# ------------------------------
# GPIO: настройки и пины
# ------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# Motor drive interface definition
ENA = LEFT_MOTOR_ENABLE = 13  # L298 Enable A
ENB = RIGHT_MOTOR_ENABLE = 12  # L298 Enable B
IN1 = LEFT_MOTOR_PIN1 = 19  # Motor interface 1
IN2 = LEFT_MOTOR_PIN2 = 16  # Motor interface 2
IN3 = RIGHT_MOTOR_PIN1 = 21  # Motor interface 3
IN4 = RIGHT_MOTOR_PIN2 = 26  # Motor interface 4

# Motor initialized to LOW
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

pwmA = GPIO.PWM(ENA, 100)
pwmB = GPIO.PWM(ENB, 100)
pwmA.start(0)
pwmB.start(0)

#------ новые

# ------------------------------
# ГЛОБАЛЬНЫЕ переменные для фильтра Калмана
# ------------------------------
X_prev = np.array([0, 0, 0, 0, 0, 0])
D_n_UWB = 0.0009
T_sample = 0.1
x_sat = np.zeros((4, 3))
D_n_mat = D_n_UWB * np.eye(4)
x_est_prev = X_prev
D_x_prev = np.eye(X_prev.shape[0])

# ------------------------------
# ФУНКЦИИ УПРАВЛЕНИЯ МОТОРАМИ
# ------------------------------

def set_motor_speeds(left_speed, right_speed):
    """Устанавливает скорости для левого и правого моторов"""
    # Левый мотор
    if left_speed > 0:
        GPIO.output(LEFT_MOTOR_PIN1, GPIO.HIGH)
        GPIO.output(LEFT_MOTOR_PIN2, GPIO.LOW)
    else:
        GPIO.output(LEFT_MOTOR_PIN1, GPIO.LOW)
        GPIO.output(LEFT_MOTOR_PIN2, GPIO.HIGH)
    pwmA.ChangeDutyCycle(abs(left_speed))

    # Правый мотор
    if right_speed > 0:
        GPIO.output(RIGHT_MOTOR_PIN1, GPIO.HIGH)
        GPIO.output(RIGHT_MOTOR_PIN2, GPIO.LOW)
    else:
        GPIO.output(RIGHT_MOTOR_PIN1, GPIO.LOW)
        GPIO.output(RIGHT_MOTOR_PIN2, GPIO.HIGH)
    pwmB.ChangeDutyCycle(abs(right_speed))

def MotorForward(speed):
    print('Движение вперед')
    set_motor_speeds(speed, speed)

def MotorBackward(speed):
    print('Движение назад')
    set_motor_speeds(-speed, -speed)

def MotorTurnRight():
    print('Поворот направо (танковое управление)')
    set_motor_speeds(-70, 70)  # Левый вперед, правый назад

def MotorTurnLeft():
    print('Поворот налево (танковое управление)')
    set_motor_speeds(70, -70)  # Левый назад, правый вперед

def MotorStop():
    print('Остановка')
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    # Сбрасываем пины в LOW состояние
    GPIO.output(LEFT_MOTOR_PIN1, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_PIN2, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_PIN1, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_PIN2, GPIO.LOW)

def RemoteControl(control):
    if control == "forward":
        MotorForward(70)
    elif control == "down":
        MotorBackward(70)
    elif control == "right":
        MotorTurnRight()
    elif control == "left":
        MotorTurnLeft()
    else:
        MotorStop()
        
# ------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------
def ClearPins():
    GPIO.cleanup()
    exit()

def on_esc():
    print("Завершение работы")
    on_close()  # Закрываем ws
    MotorStop()


def Kill(proc_pid):
    try:
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
    except Exception as e:
        print("Ошибка при завершении процесса:", e)

def Encode(code):
    message_bytes = code.encode('ascii')
    base64_bytes = base64.b64encode(message_bytes)
    return base64_bytes.decode('ascii')

def Decode(base64_message):
    base64_bytes = base64_message.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)
    return message_bytes.decode()

# ------------------------------
# ФУНКЦИИ ФИЛЬТРА КАЛМАНА
# ------------------------------
def uwb_kalman(log_str):
    global D_x_prev, x_est_prev, x_sat
    store = reader_logs(log_str)
    list_keys = []
    j = 0
    try:
        for key in store:
            list_keys.append(key)
            x_sat[j] = store[key]['x_sat']
            j += 1
        R = {key: store[key]['range'] for key in store}
        x_est, D_x = EKF3(R, x_est_prev, D_x_prev, D_n_mat, x_sat, T_sample, list_keys)
        D_x_prev = D_x
        x_est_prev = x_est
        return x_est[0] * 100, x_est[1] * 100
    except Exception as e:
        return None, None

# ------------------------------
# ЛОКАЛЬНЫЙ WebSocket-СЕРВЕР (для локальных клиентов)
# ------------------------------
local_ws_clients = set()
local_loop = None

async def local_server(websocket, path):
    global local_ws_clients
    local_ws_clients.add(websocket)
    print("Локальный клиент подключился:", websocket.remote_address)
    try:
        async for message in websocket:
            pass
    except websockets.exceptions.ConnectionClosedError:
        print("Локальный клиент отключился:", websocket.remote_address)
    finally:
        local_ws_clients.remove(websocket)

def run_local_ws_server():
    global local_loop
    local_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(local_loop)
    server = websockets.serve(local_server, "localhost", 8765)
    local_loop.run_until_complete(server)
    print("Локальный WebSocket сервер запущен на localhost:8765")
    local_loop.run_forever()

def broadcast_to_local_clients(message):
    async def _broadcast():
        for client in list(local_ws_clients):
            try:
                await client.send(message)
            except Exception as e:
                print("Ошибка отправки локальному клиенту:", e)
    if local_loop:
        asyncio.run_coroutine_threadsafe(_broadcast(), local_loop)

# ------------------------------
# ФУНКЦИИ ДЛЯ ОТПРАВКИ СООБЩЕНИЙ ЧЕРЕЗ WebSocket
# ------------------------------
def sendPos(position):
    global api
    j_mes = json.dumps({
        "apikey": api,
        "action": "send_telemetry",
        "payload": {
            "x": position[0],
            "y": position[1],
            "w_x": position[2],
            "w_y": position[3],
            "w_z": position[4]
        }
    })
    safe_ws_send(j_mes)

def sendLog(message):
    global api
    j_mes = json.dumps({
        "apikey": api,
        "action": "send_log",
        "payload": {"message": message}
    })
    safe_ws_send(j_mes)

def Login():
    j_mes = json.dumps({
        "action": "device_login",
        "payload": {"name": login, "password": password}
    })
    safe_ws_send(j_mes)

def Close():
    global api
    j_mes = json.dumps({
        "apikey": api,
        "action": "send_log",
        "payload": {"name": login}
    })
    safe_ws_send(j_mes)

def safe_ws_send(message):
    """
    Отправка сообщения через WebSocket с обработкой ошибок.
    Если соединение закрыто, сообщение не отправляется и выводится предупреждение.
    """
    global ws
    try:
        if ws:
            ws.send(message)
        else:
            print("Ошибка: WebSocket-соединение отсутствует. Сообщение не отправлено.")
    except Exception as e:
        print("Ошибка при отправке через WebSocket:", e)

# ------------------------------
# WebRTC: Класс для передачи видео по WebRTC (робот – отвечающая сторона)
# ------------------------------
class VideoCaptureTrack(VideoStreamTrack):
    """
    Видео-трек, захватывающий кадры с камеры через OpenCV.
    """
    def __init__(self):
        super().__init__()
        i = cam_check()
        if i is None:
            print('Нет доступных камер')
            return
        self.cap = cv2.VideoCapture(i)
        if not self.cap.isOpened():
            print("Не удалось открыть камеру")
            return
        # Устанавливаем разрешение, чтобы избежать лишнего масштабирования
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Преобразуем BGR (OpenCV) в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        cv2.waitKey(0)
        return video_frame


class WebRTCClient:
    def __init__(self):
        self.api_key = ""
        self.loop = asyncio.new_event_loop()
        self.pc = RTCPeerConnection(RTCConfiguration(
        ))
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def handle_offer(self, sdp, ws_app):
        
        print("SDP---------", sdp)
        
        video_track = VideoCaptureTrack()
        # Добавляем трансивер с явным направлением "sendonly"
        transceiver = self.pc.addTransceiver("video", direction="sendonly")
        transceiver.sender.replaceTrack(video_track)
        
        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                candidate_msg = json.dumps({
                    "action": "candidate",
                    "type": "candidate",
                    "apikey":  api,
                    "payload": {
                        "sdp": {
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex
                        }
                    }
                })
                # Исправлено: вместо websocket.send используем ws_app.send для отправки кандидата
                await ws_app.send(candidate_msg)
        
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        response = json.dumps({
            "action": "answer",
            "type": "answer",
            "apikey": api,
            "payload": {
                "sdp": self.pc.localDescription.sdp
            }
        })
        await ws_app.send(response)
        print("Отправлен SDP ответ клиенту.")
        

    async def add_candidate(self, candidate_payload):
        try:
            candidate_json = candidate_payload.get("sdp", "{}")
                    
            candidate = json.loads(candidate_json)
            print(candidate)
            ip = candidate['candidate'].split(' ')[4]
            port = candidate['candidate'].split(' ')[5]
            protocol = candidate['candidate'].split(' ')[7]
            priority = candidate['candidate'].split(' ')[3]
            foundation = candidate['candidate'].split(' ')[0]
            component = candidate['candidate'].split(' ')[1]
            type_ = candidate['candidate'].split(' ')[7]
            ice_candidate = RTCIceCandidate(
                ip=ip,
                port=port,
                protocol=protocol,
                priority=priority,
                foundation=foundation,
                component=component,
                type=type_,
                sdpMid=candidate['sdpMid'],
                sdpMLineIndex=candidate['sdpMLineIndex']
            )
            await self.pc.addIceCandidate(ice_candidate)
            print("Добавлен ICE candidate:", ice_candidate)
        except Exception as e:
            print("Ошибка при добавлении ICE candidate:", e)

    async def close(self):
        await self.pc.close()
        self.loop.stop()

# ------------------------------
# ОБРАБОТКА ВХОДЯЩИХ СООБЩЕНИЙ (единое соединение WebSocket)
# ------------------------------
def on_message(ws_app, message):
    global api, stop, car_id, user_id, status_connection, All_cars, webrtc_client
    try:
        jmes = json.loads(message)
    except Exception as e:
        print("Ошибка парсинга JSON:", e)
        return
    print(jmes)
    action = jmes.get("action", "")

    if action == "device_login":
        api = jmes["data"]["api_key"]
        car_id = jmes["data"]["car_id"]
        print("Логин прошёл успешно, api:", api, "car_id:", car_id)

    elif action == "offer":
        if webrtc_client is None:
            webrtc_client = WebRTCClient()
        webrtc_client.api_key = api
        sdp = jmes["payload"]["sdp"]
        print("sdp", sdp)
        asyncio.run_coroutine_threadsafe(webrtc_client.handle_offer(sdp, ws_app), webrtc_client.loop)

    elif action == "candidate":
        if webrtc_client is not None:
            asyncio.run_coroutine_threadsafe(
                webrtc_client.add_candidate(jmes["payload"]),
                webrtc_client.loop
            )
        else:
            print("WebRTCClient ещё не инициализирован")

    elif action == "car_selected":
        user_id = jmes["payload"]["user_id"]
        status_connection = True

    elif action == "car_released":
        if jmes["payload"]["user_id"] == user_id:
            MotorStop()
            status_connection = False
            if webrtc_client is not None:
                asyncio.run_coroutine_threadsafe(webrtc_client.close(), webrtc_client.loop)
                webrtc_client = None

    elif action == "send_command":
        cmd = jmes["payload"]["command"]
        if cmd == "forward":
            RemoteControl("forward")
        elif cmd == "backward":
            RemoteControl("down")
        elif cmd == "left":
            RemoteControl("left")
        elif cmd == "right":
            RemoteControl("right")
        else:
            RemoteControl("stop")

    elif action == "car_position":
        All_cars.append({
            "car_id": jmes["payload"]["car_id"],
            "role": jmes["payload"]["role"],
            "x": jmes["payload"]["x"],
            "y": jmes["payload"]["y"],
            "w_x": jmes["payload"]["w_x"],
            "w_y": jmes["payload"]["w_y"],
            "w_z": jmes["payload"]["w_z"]
        })
        broadcast_to_local_clients(json.dumps({
            "action": "SendAllCoords",
            "data": {"data": jmes["data"].get("localDescription", ""), "position": jmes["payload"]}
        }))

    elif action == "send_file":
        print("Получение файла")
        s = jmes["payload"]["body"]
        ext = jmes["payload"]["extension"]
        filename = "user.py" if ext == "py" else "any_file." + ext
        with open(filename, "w") as f:
            f.write(Decode(s))
        print("Файл получен:", filename)

    elif action == "run_program":
        print("Запуск пользовательского процесса...")
        new = Thread(target=NewProc)
        new.start()
        stop = 0
    elif action == "stop_program":
        print("Остановка процесса")
        stop = 1
        if proc.poll() is None:
            Kill(proc.pid)
            _, err = proc.communicate()
            if err:
                sendLog(err)
        else:
            _, err = proc.communicate()
            sendLog(err if err else "No error")
        sendLog("close")
        MotorStop()
    else:
        print("Неизвестное действие:", action)

def on_close(ws_app=None, close_status_code=None, close_msg=None):
    global ws
    print("WebSocket соединение закрыто.")
    ws = None

def on_open(ws_app):
    print("Соединение установлено")
    sleep(2)
    Login()

def on_error(ws_app, error):
    print("Ошибка:", error)

# ------------------------------
# Запуск пользовательского кода
# ------------------------------
def NewProc():
    print("Запуск user.py")
    global proc, stop
    proc = subprocess.Popen(["python", "user.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while stop == 0 and proc.poll() is None:
        sleep(1)
        instr = proc.stdout.readline()
        if instr != '':
            print(instr)
            sendLog(instr)
    if proc.poll() is None:
        Kill(proc.pid)
        _, err = proc.communicate()
        if err:
            sendLog(err)
    else:
        _, err = proc.communicate()
        sendLog(err if err else "No error")
    sendLog("close")
    MotorStop()

# ------------------------------
# Асинхронное чтение данных датчика (UWB)
# ------------------------------
async def read_sensor_data(port='/dev/ttyACM0', baudrate=115200, reconnect_interval=5, max_no_data_time=5):
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = '/dev/ttyACM0'
    ser.open()
    ser.timeout = 1
    ser.write(b'\r\r')
    data = ser.readline()
    ser.write(b'les\n')
    while True:
        try:
            ser.write(b'\r\r')
            data = ser.readline()
            ser.write(b'les\n')
            print("Подключение к UWB датчику установлено")
            sendLog("Подключение к UWB датчику установлено")
            last_valid_data_time = time.time()
            while True:
                if ser.in_waiting:
                    new_str = ser.readline().decode('ascii').strip()
                    if len(new_str) > 10:
                        log = str(new_str)
                        x_, y_ = uwb_kalman(log)
                        if x_ is not None:
                            x_out, y_out = x_, y_
                            x, y, z = 0, 0, 0  # Заглушка для углов
                            broadcast_to_local_clients(json.dumps({
                                "action": "SendPos",
                                "position": [x_out / 100, y_out / 100, x, y, z]
                            }))
                            if status_connection:
                                sendPos([x_out / 100, y_out / 100, x, y, z])
                            last_valid_data_time = time.time()
                    else:
                        print("Получены неверные данные")
                if time.time() - last_valid_data_time > max_no_data_time:
                    print("Долгое отсутствие валидных данных, переподключение датчика...")
                    ser.close()
                    ser.open()
                    break
                await asyncio.sleep(0.1)
        except (serial.SerialException, OSError) as e:
            print(f"Ошибка соединения с датчиком: {e}, повтор через {reconnect_interval} сек...")
        await asyncio.sleep(reconnect_interval)

def run_sensor_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(read_sensor_data())


def cam_check():
    cam_test = 20
    for i in range(cam_test):
        try:
            cap = cv2.VideoCapture(i)
            test, frame = cap.read()
            if test:
                cap.release()
                return i
            cap.release()
        except:
            pass
    return None

atexit.register(on_esc)

# ------------------------------
# ОСНОВНОЕ СОЕДИНЕНИЕ С УДАЛЁННЫМ WebSocket-СЕРВЕРОМ
# ------------------------------
ssl_options = {
    "cert_reqs": ssl.CERT_NONE,
    "ssl_version": ssl.PROTOCOL_TLSv1_2
}

def run_ws():
    global ws
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://194.67.86.110:9001/robot",
                on_message=on_message,
                on_open=on_open,
                on_close=on_close,
                on_error=on_error
            )
            ws.run_forever(sslopt=ssl_options, ping_interval=30, ping_timeout=10)
        except Exception as e:
            print("Ошибка в основном WebSocket потоке:", e)
        print("Соединение потеряно. Переподключаемся через 10 секунд...")
        time.sleep(10)

# ------------------------------
# Запуск всех потоков
# ------------------------------
if __name__ == "__main__":
    # Поток соединения с удалённым WebSocket-сервером
    ws_thread = Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # Поток локального WebSocket-сервера
    local_ws_thread = Thread(target=run_local_ws_server, daemon=True)
    local_ws_thread.start()

    # Поток чтения данных датчика
    sensor_thread = Thread(target=run_sensor_loop, daemon=True)
    sensor_thread.start()

    # Главный поток ожидания завершения
    while True:
        time.sleep(1)
