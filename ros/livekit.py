import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import asyncio
import websockets
import json
import threading
import ssl
import certifi
import traceback
from datetime import datetime

SERVER_URI = "wss://194.67.86.110:9001/apriltag"
AUTH_PAYLOAD = {"action": "apriltag_system_login", "payload": {"name": "dima_tag", "password": "jopapopa"}}
SEND_INTERVAL = 0.3  # секунда между отправками координат

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

class WSNode(Node):
    def __init__(self):
        super().__init__('ws_node')

        # Публикатор для токена LiveKit
        self.token_pub = self.create_publisher(String, 'livekit/token', 10)
        self.user_code_pub = self.create_publisher(String, 'user/code', 10)

        # Подписка на координаты
        self.create_subscription(Float32MultiArray, 'apriltag/coords', self.coord_callback, 10)

        # Последние координаты
        self.latest_coords = None

        # Данные из сервера
        self.api_key = None
        self.layout_id = None
        self.livekit_token = None
        self.user_code = None
        self.car_tag_mapping = {}

        # Запускаем asyncio loop в отдельном потоке
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self.start_loop, daemon=True)
        t.start()

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.websocket_loop())

    def coord_callback(self, msg):
        """Сохраняем только последние координаты"""
        self.latest_coords = list(msg.data)

    async def websocket_loop(self):
        while True:
            try:
                self.get_logger().info(f"Connecting to server {SERVER_URI} ...")
                async with websockets.connect(SERVER_URI, ssl=ssl_context, ping_interval=20, ping_timeout=10) as ws:
                    self.get_logger().info("Connected to server")
                    await ws.send(json.dumps(AUTH_PAYLOAD))

                    auth_resp_str = await asyncio.wait_for(ws.recv(), timeout=10)
                    auth_resp = json.loads(auth_resp_str)
                    if auth_resp.get("status") != "success":
                        self.get_logger().error(f"Auth failed: {auth_resp}")
                        await asyncio.sleep(5)
                        continue

                    data = auth_resp.get("data", {})
                    self.api_key = data.get("api_key")
                    self.layout_id = data.get("layout_id")
                    self.livekit_token = data.get("livekit_token")
                    if self.livekit_token:
                        self.token_pub.publish(String(data=self.livekit_token))

                    # car_tag_mapping
                    self.car_tag_mapping.clear()
                    for car_conf in data.get("cars_config", []):
                        car_id = car_conf.get("car_id")
                        tag_id = car_conf.get("assigned_tag_id")
                        if car_id and tag_id is not None:
                            self.car_tag_mapping[str(tag_id)] = car_id

                    # Запуск задач: приём сообщений и отправка координат
                    recv_task = asyncio.create_task(self.receive_server(ws))
                    send_task = asyncio.create_task(self.send_coords(ws))
                    done, pending = await asyncio.wait(
                        [recv_task, send_task],
                        return_when=asyncio.FIRST_EXCEPTION
                    )

                    for t in pending:
                        t.cancel()

            except Exception as e:
                self.get_logger().error(f"WebSocket error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

    async def receive_server(self, ws):
        try:
            async for msg_str in ws:
                msg = json.loads(msg_str)
                action = msg.get("action")
                if action == "send_file_to_apriltag":
                    self.user_code = msg.get("payload", {}).get("body")
                    if self.user_code:
                        self.user_code_pub.publish(String(data=self.user_code))
                elif action == "layout_config_updated":
                    temp_mapping = {}
                    for car_conf in msg.get("payload", {}).get("cars_config", []):
                        car_id = car_conf.get("car_id")
                        tag_id = car_conf.get("assigned_tag_id")
                        if car_id and tag_id is not None:
                            temp_mapping[str(tag_id)] = car_id
                    self.car_tag_mapping = temp_mapping
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().warn("Server connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.get_logger().error(f"Error in receive_server: {e}")
            traceback.print_exc()

    async def send_coords(self, ws):
        """Отправляем только последние координаты каждые SEND_INTERVAL секунд"""
        try:
            while True:
                if not self.latest_coords or not self.api_key or not self.layout_id:
                    await asyncio.sleep(0.05)
                    continue
                try:
                    tag_id = str(int(self.latest_coords[0]))
                    if tag_id not in self.car_tag_mapping:
                        await asyncio.sleep(0.05)
                        continue

                    tags_data = [{
                        "tag_id": tag_id,
                        "car_id": self.car_tag_mapping[tag_id],
                        "x": self.latest_coords[1],
                        "y": self.latest_coords[2],
                        "z": self.latest_coords[3],
                        "vx": self.latest_coords[4],
                        "vy": self.latest_coords[5],
                        "vz": self.latest_coords[6],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    }]

                    payload = {
                        "action": "april_tags_data",
                        "apikey": self.api_key,
                        "payload": {
                            "layout_id": self.layout_id,
                            "tags": tags_data
                        }
                    }

                    await ws.send(json.dumps(payload))
                    await asyncio.sleep(SEND_INTERVAL)
                except Exception as e:
                    self.get_logger().error(f"Error sending coords: {e}")
        except asyncio.CancelledError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = WSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
