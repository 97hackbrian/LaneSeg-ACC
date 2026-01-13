#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo

class CameraBufferNode(Node):
    def __init__(self):
        super().__init__('camera_buffer_node')

        # --- CONFIGURACIÓN ---
        # El tópico "raro" o real de tu cámara
        self.input_topic = '/camera/color_image'
        
        # Los nuevos tópicos "limpios" que usará Isaac ROS
        self.output_image_topic = '/buffer/image'
        self.output_info_topic = '/buffer/camera_info'
        
        self.img_width = 640
        self.img_height = 480
        # ---------------------

        # 1. Configurar QoS de entrada "permisivo" (acepta todo)
        # Esto ayuda si tu cámara es Best Effort
        qos_input = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. Configurar QoS de salida "estricto" (lo que le gusta a Isaac ROS)
        qos_output = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Suscriptor (Escucha a tu cámara real)
        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_input # Usa el perfil permisivo
        )

        # Publicadores (Generan los tópicos limpios)
        self.pub_image = self.create_publisher(Image, self.output_image_topic, qos_output)
        self.pub_info = self.create_publisher(CameraInfo, self.output_info_topic, qos_output)

        self.get_logger().info(f"Buffer iniciado. Escuchando: {self.input_topic}")
        self.get_logger().info(f"Publicando buffers en: {self.output_image_topic} y {self.output_info_topic}")

    def image_callback(self, msg):
        # 1. Crear el mensaje de CameraInfo artificial
        info_msg = CameraInfo()
        
        # CRÍTICO: Copiar el header exacto de la imagen original
        # Esto engaña a Isaac ROS para que crea que están perfectamente sincronizados
        info_msg.header = msg.header 
        
        info_msg.height = self.img_height
        info_msg.width = self.img_width
        info_msg.distortion_model = "plumb_bob"
        
        # Matriz K (Intrínseca) dummy pero válida
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        fx = 500.0
        fy = 500.0
        info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] # Sin distorsión
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        # 2. Publicar ambos (La imagen original y la info generada)
        self.pub_image.publish(msg)
        self.pub_info.publish(info_msg)

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