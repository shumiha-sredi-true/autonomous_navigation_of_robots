# LiveKitNode
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import asyncio
import cv2
from livekit import rtc
import time

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LIVEKIT_URL = "ws://194.67.86.110:7880"

class LiveKitNode(Node):
    def __init__(self):
        super().__init__('livekit_node')
        self.bridge = CvBridge()
        self.token = None
        self.room = None
        self.video_source = None
        self.track = None
        self.loop = asyncio.get_event_loop()

        self.create_subscription(String, '/livekit/token', self.token_callback, 10)
        self.create_subscription(Image, 'tech_v/image_raw', self.image_callback, 10)

    def token_callback(self, msg):
        if self.token is None:
            self.token = msg.data
            self.get_logger().info(f"Received LiveKit token, connecting...")
            self.loop.create_task(self.connect_room())

    async def connect_room(self):
        self.room = rtc.Room()
        await self.room.connect(LIVEKIT_URL, self.token)
        self.video_source = rtc.VideoSource(FRAME_WIDTH, FRAME_HEIGHT)
        self.track = rtc.LocalVideoTrack.create_video_track("tech_v", self.video_source)
        await self.room.local_participant.publish_track(self.track)
        self.get_logger().info("LiveKit connected!")

    def image_callback(self, msg):
        if self.video_source is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vf = rtc.VideoFrame(FRAME_WIDTH, FRAME_HEIGHT, rtc.VideoBufferType.RGB24, frame_rgb.tobytes())
        ts = int(time.time() * 1e6)
        self.video_source.capture_frame(vf, timestamp_us=ts)
