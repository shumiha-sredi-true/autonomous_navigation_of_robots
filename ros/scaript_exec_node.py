import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import asyncio
import websockets
import json
import traceback

SERVER_URI = "wss://194.67.86.110:9001/script_exec"

class ScriptExecNode(Node):
    def __init__(self):
        super().__init__('script_exec_node')
        self.subscription = self.create_subscription(Image, 'tech_v/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.listen_server())

    def image_callback(self, msg):
        # Обновляем latest_frame для скриптов
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    async def listen_server(self):
        while True:
            try:
                async with websockets.connect(SERVER_URI, ping_interval=20, ping_timeout=10) as websocket:
                    print("Connected to script server")
                    async for message in websocket:
                        try:
                            # Получаем скрипт и выполняем его
                            exec_globals = {'frame': self.latest_frame}
                            exec(message, exec_globals)
                        except Exception as e:
                            print(f"Script execution error: {e}")
                            traceback.print_exc()
            except Exception as e:
                print(f"WebSocket error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

def main(args=None):
    rclpy.init(args=args)
    node = ScriptExecNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
