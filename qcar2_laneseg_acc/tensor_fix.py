#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from isaac_ros_tensor_list_interfaces.msg import TensorList, TensorShape

class TensorFixNode(Node):
    def __init__(self):
        super().__init__('tensor_fix_node')

        # --- PERFIL 1: Entrada (Permisivo) ---
        # Aceptamos lo que sea que mande TensorRT
        qos_input = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- PERFIL 2: Salida (Estricto) ---
        # EL DECODER EXIGE ESTO. Si no es RELIABLE, rechaza la conexión.
        qos_output = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 1. Suscripción (Desde TensorRT)
        self.sub = self.create_subscription(
            TensorList, 
            '/tensor_raw_output', 
            self.callback, 
            qos_input
        )
        
        # 2. Publicación (Hacia Decoder) -> Usamos el perfil RELIABLE
        self.pub = self.create_publisher(
            TensorList, 
            '/tensor_decoder_input', 
            qos_output
        )
        
        self.get_logger().info("--- PUENTE DE FUERZA BRUTA ACTIVO (QoS: Reliable) ---")

    def callback(self, msg):
        if not msg.tensors: return

        tensor = msg.tensors[0]
        d = tensor.shape.dims 

        # LOG DE CONTROL (Solo para ver si fluye)
        # self.get_logger().info(f"Procesando: {d}")

        # --- APLICAR CORRECCIÓN INCONDICIONAL ---
        # Convertimos NCHW [1, 1, 256, 256] -> NHWC [1, 256, 256, 1]
        
        new_shape = TensorShape()
        new_shape.rank = 4
        # Mover dimensiones: N=0, H=2, W=3, C=1
        new_shape.dims = [d[0], d[2], d[3], d[1]] 
        tensor.shape = new_shape
            
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TensorFixNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()