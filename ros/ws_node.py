import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import asyncio
import websockets
import json

SERVER_URI = "wss://194.67.86.110:9001/apriltag"
AUTH_PAYLOAD = {"action": "apriltag_system_login","payload": {"name": "dima_tag", "password": "jopapopa"}}

class WSNode(Node):
    def __init__(self):
        super().__init__('ws_node')
        self.subscription = self.create_subscription(Float32MultiArray, 'apriltag/coords', self.coord_callback, 10)
        self.websocket = None
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.connect_ws())

    async def connect_ws(self):
        self.websocket = await websockets.connect(SERVER_URI, ping_interval=20, ping_timeout=10)
        await self.websocket.send(json.dumps(AUTH_PAYLOAD))
        print("Authenticated with server")

    def coord_callback(self, msg):
        if self.websocket is None:
            return
        data = {"action": "april_tags_data", "payload": msg.data.tolist()}
        asyncio.run_coroutine_threadsafe(self.websocket.send(json.dumps(data)), self.loop)

def main(args=None):
    rclpy.init(args=args)
    node = WSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
