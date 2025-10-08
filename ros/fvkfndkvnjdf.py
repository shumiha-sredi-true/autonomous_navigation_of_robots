import rclpy
from rclpy.node import Node
from rclpy.executors import AsyncIOExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import asyncio
import cv2
import time
from livekit import rtc

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECONNECT_DELAY = 5  # секунд между попытками подключения


class LiveKitNode(Node):
    def __init__(self):
        super().__init__('livekit_node')

        self.bridge = CvBridge()
        self.token = None
        self.livekit_url = None
        self.room = None
        self.video_source = None
        self.track = None
        self.connected = False

        # Асинхронные подписки
        self.create_subscription(String, 'livekit/token', self.token_callback, 10)
        self.create_subscription(Image, 'process/image', self.image_callback, 10)

        self.get_logger().info("LiveKit node initialized, waiting for token...")

    async def token_callback(self, msg: String):
        """Асинхронный коллбэк получения токена"""
        try:
            self.livekit_url, self.token = msg.data.split("|", 1)
            self.get_logger().info(f"Received token and URL: {self.livekit_url}")
            # Запускаем асинхронную задачу подключения
            asyncio.create_task(self.connect_room_loop())
        except Exception as e:
            self.get_logger().error(f"Error parsing token: {e}")

    async def connect_room_loop(self):
        """Цикл подключения к LiveKit с повторными попытками"""
        delay = 1
        while True:
            if self.token is None:
                await asyncio.sleep(0.5)
                continue

            try:
                await self.connect_room()
                self.connected = True
                self.get_logger().info("✅ Successfully connected to LiveKit")
                return
            except Exception as e:
                self.get_logger().error(f"Failed to connect to LiveKit: {e}")
                self.get_logger().info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)  # экспоненциальная задержка

    async def connect_room(self):
        """Подключение к комнате и публикация видео"""
        self.get_logger().info(f"Connecting to LiveKit at {self.livekit_url}...")
        self.room = rtc.Room()
        await self.room.connect(self.livekit_url, self.token)
        self.video_source = rtc.VideoSource(FRAME_WIDTH, FRAME_HEIGHT)
        self.track = rtc.LocalVideoTrack.create_video_track("camera", self.video_source)
        await self.room.local_participant.publish_track(self.track)
        self.get_logger().info("✅ LiveKit connected!")

    async def image_callback(self, msg: Image):
        """Отправка кадров в LiveKit"""
        if not self.connected or self.video_source is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vf = rtc.VideoFrame(
                FRAME_WIDTH,
                FRAME_HEIGHT,
                rtc.VideoBufferType.RGB24,
                frame_rgb.tobytes()
            )
            ts = int(time.time() * 1e6)  # микросекунды
            self.video_source.capture_frame(vf, timestamp_us=ts)
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LiveKitNode()

    # AsyncIOExecutor для работы с async коллбэками
    executor = AsyncIOExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down LiveKitNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
