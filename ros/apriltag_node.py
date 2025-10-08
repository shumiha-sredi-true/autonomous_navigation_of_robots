import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

TAG_SIZE = 0.04
CALIBRATION_TAG_ID = 13

camera_matrix = np.array([[1.44464602e03, 0, 9.06452518e02],
                          [0, 1.43108865e03, 4.68944978e02],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.172642475, -0.440297855, 1.11232097e-04, 5.56096772e-04, 0.504451308]])

class AprilTagNode(Node):
    def __init__(self):
        super().__init__('apriltag_node')
        self.subscription = self.create_subscription(Image, 'tech_v/image_raw', self.callback, 10)
        self.publisher_ = self.create_publisher(Float32MultiArray, 'apriltag/coords', 10)
        self.bridge = CvBridge()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        self.R_calib_matrix = None
        self.t0_calib_vector = None
        self.last_positions = {}

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return

        for i, tag_id_array in enumerate(ids):
            tag_id = int(tag_id_array[0])
            marker_corners = corners[i][0]
            obj_points = np.array([[-TAG_SIZE/2, TAG_SIZE/2, 0],
                                   [TAG_SIZE/2, TAG_SIZE/2, 0],
                                   [TAG_SIZE/2, -TAG_SIZE/2, 0],
                                   [-TAG_SIZE/2, -TAG_SIZE/2, 0]], dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners.astype(np.float32), camera_matrix, dist_coeffs)
            if not success:
                continue

            tvec_flat = tvec.flatten()
            if tag_id == CALIBRATION_TAG_ID and (self.R_calib_matrix is None or self.t0_calib_vector is None):
                R_temp, _ = cv2.Rodrigues(rvec)
                self.R_calib_matrix = R_temp
                self.t0_calib_vector = tvec_flat
                continue

            if self.R_calib_matrix is not None and self.t0_calib_vector is not None:
                tvec_calib = self.R_calib_matrix.T @ (tvec_flat - self.t0_calib_vector)
                now = time.time()
                velocity = np.array([0.,0.,0.])
                if tag_id in self.last_positions:
                    prev = self.last_positions[tag_id]
                    dt = now - prev["time"]
                    if dt > 0:
                        velocity = (tvec_calib - prev["pos"]) / dt
                self.last_positions[tag_id] = {"pos": tvec_calib.copy(), "time": now, "vel": velocity.copy()}

                msg_out = Float32MultiArray(data=[tag_id, *tvec_calib, *velocity])
                self.publisher_.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
