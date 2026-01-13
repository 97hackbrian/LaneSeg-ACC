#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
import cv2
import numpy as np

class CameraBufferNode(Node):
    def __init__(self):
        super().__init__('camera_buffer_node')

        # --- CONFIGURACIÓN ---
        self.input_topic = '/camera/color_image'
        self.output_image_topic = '/buffer/image_resized'
        self.output_info_topic = '/buffer/camera_info'
        
        # Dimensiones estrictas para Isaac ROS
        self.target_width = 512
        self.target_height = 512
        # ---------------------

        qos_input = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_output = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_input
        )

        self.pub_image = self.create_publisher(Image, self.output_image_topic, qos_output)
        self.pub_info = self.create_publisher(CameraInfo, self.output_info_topic, qos_output)

        self.get_logger().info(f"Buffer (512x512) listo. Esperando imágenes...")

    def image_callback(self, msg):
        try:
            # --- 1. VALIDACIÓN DE ENTRADA ---
            if msg.width == 0 or msg.height == 0 or len(msg.data) == 0:
                return # Ignorar frames vacíos de inicialización

            pixel_count = msg.width * msg.height
            channels = int(len(msg.data) / pixel_count)

            # --- 2. PROCESAMIENTO DE IMAGEN ---
            # Recuperar imagen raw
            np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)

            # Normalizar canales a 3 (BGR)
            if channels == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2BGR)
            elif channels == 1:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)

            # Resize a 512x512
            resized = cv2.resize(np_img, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

            # Asegurar memoria contigua (Vital para evitar SegFaults)
            if not resized.flags['C_CONTIGUOUS']:
                resized = np.ascontiguousarray(resized)

            # --- 3. CREAR MENSAJE DE IMAGEN ---
            out_img = Image()
            out_img.header = msg.header
            out_img.height = self.target_height
            out_img.width = self.target_width
            out_img.encoding = 'bgr8'
            out_img.is_bigendian = 0
            out_img.step = self.target_width * 3
            out_img.data = resized.tobytes()

            # --- 4. CREAR CAMERA INFO (SOLUCIÓN DEL ERROR ACTUAL) ---
            info_msg = CameraInfo()
            info_msg.header = msg.header
            info_msg.height = self.target_height
            info_msg.width = self.target_width
            
            # ESTO FALTABA: Definir el modelo de distorsión explícitamente
            info_msg.distortion_model = "plumb_bob"
            
            # Generar matriz Intrínseca (K) dummy centrada para 512x512
            cx = self.target_width / 2.0
            cy = self.target_height / 2.0
            fx = 500.0 # Focal length aproximada
            fy = 500.0

            # K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            
            # D: Distorsión cero
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # R: Identidad (sin rotación)
            info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            
            # P: Proyección [fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0]
            info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

            # --- 5. PUBLICAR ---
            self.pub_image.publish(out_img)
            self.pub_info.publish(info_msg)

        except Exception as e:
            self.get_logger().error(f"Error procesando frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()