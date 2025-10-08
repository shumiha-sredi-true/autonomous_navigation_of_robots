import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher = self.create_publisher(Image, '/tech_v/image_raw', 10)
        self.bridge = CvBridge()

        # параметры: индекс камеры и частота публикации
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('fps', 30)

        cam_index = self.get_parameter('camera_index').value
        self.fps = self.get_parameter('fps').value

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Не удалось открыть камеру с индексом {cam_index}')

        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Не удалось получить кадр с камеры')
            return

        # OpenCV использует BGR, ROS обычно ожидает BGR8
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)
        self.get_logger().debug('Опубликован кадр')

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
