import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import base64
import numpy as np


class NodeManager(Node):
    def __init__(self):
        super().__init__('node_manager')

        # Подписка на JSON от WebSocket-ноды
        self.create_subscription(String, '/ws_messages', self.ws_callback, 10)

        # Подписка на камеру
        self.bridge = CvBridge()
        self.create_subscription(Image, '/tech_v/image_raw', self.camera_callback, 10)

        # Публикация обработанных кадров для LiveKit
        self.publisher_ = self.create_publisher(Image, '/processed_video', 10)

        # Дефолтная функция обработки
        self.process_frame = lambda frame: frame

        # Имя функции, которое NodeManager ищет
        self.function_name = "process_frame"

        self.get_logger().info("NodeManager запущен")

    def ws_callback(self, msg: String):
        """Обработка JSON с кодом и именем функции"""
        try:
            data = json.loads(msg.data)
            action = data.get("action")
            payload = data.get("payload", {})

            if action in ["set_processing", "send_file_to_apriltag"]:
                self.function_name = payload.get("function_name", "process_frame")
                self.load_processing_script(payload.get("body"))

        except Exception as e:
            self.get_logger().error(f"Ошибка обработки WS-сообщения: {e}")

    def load_processing_script(self, code_b64: str):
        """Проверка скрипта: синтаксис, наличие функции, тест на dummy-кадре"""
        if not code_b64:
            self.get_logger().warning("Нет кода для обработки")
            return

        try:
            code = base64.b64decode(code_b64).decode("utf-8")
            compile(code, "<dynamic>", "exec")  # проверка синтаксиса

            local_vars = {}
            exec(code, {}, local_vars)

            if self.function_name not in local_vars:
                self.get_logger().warning(f"Функция '{self.function_name}' не найдена в скрипте")
                return

            func = local_vars[self.function_name]
            if not callable(func):
                self.get_logger().warning(f"'{self.function_name}' не является функцией")
                return

            # Тестовый прогон на dummy-кадре
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            result = func(dummy)
            if not isinstance(result, np.ndarray):
                self.get_logger().warning(f"{self.function_name} вернула не numpy.ndarray")
                return

            self.process_frame = func
            self.get_logger().info(f"Функция '{self.function_name}' успешно загружена и проверена")

        except SyntaxError as e:
            self.get_logger().error(f"Ошибка синтаксиса: {e}")
        except Exception as e:
            self.get_logger().error(f"Ошибка при загрузке скрипта: {e}")

    def camera_callback(self, msg: Image):
        """Обработка каждого кадра камеры и публикация"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            processed = self.process_frame(cv_image)
            if not isinstance(processed, np.ndarray):
                self.get_logger().warning("process_frame вернул некорректный тип")
                return
            out_msg = self.bridge.cv2_to_imgmsg(processed, encoding='bgr8')
            self.publisher_.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Ошибка выполнения process_frame: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = NodeManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
