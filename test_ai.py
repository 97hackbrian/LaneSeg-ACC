import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os

class QCarAI(Node):
    def __init__(self):
        super().__init__('qcar_ai_node')

        # 1. Cargar modelo
        # Asegúrate de que 'road_model.pt' esté en la misma carpeta donde ejecutas el script
        model_path = './road_model.pt'

        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ ERROR: No encuentro el modelo en: {model_path}")
            # Es buena práctica cerrar si no hay modelo, pero por ahora lo dejamos correr
            
        self.get_logger().info(f'Cargando cerebro YOLO desde: {model_path} ...')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('✅ ¡Cerebro cargado y listo!')
        except Exception as e:
            self.get_logger().error(f"❌ Error fatal cargando YOLO: {e}")

        # 2. Suscribirse a la cámara (AJUSTADO EL TÓPICO)
        # Nota: Si no recibes imagen, verifica el tópico con 'ros2 topic list'
        self.subscription = self.create_subscription(
            Image,
            '/camera/csi_image', 
            self.image_callback,
            10)

        # 3. Publicar visión (Debug)
        self.publisher = self.create_publisher(Image, '/qcar/ai_debug', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # --- INICIO DE LA CORRECCIÓN DE INDENTACIÓN ---
            
            # Inferencia (Bajé la confianza a 0.4 para probar)
            results = self.model.predict(frame, conf=0.4, verbose=False)
            
            # Dibujar las cajas/máscaras en la imagen
            annotated_frame = results[0].plot()
            
            # Aquí es donde iría la lógica de control (Dirección y Velocidad)
            # Por ejemplo: calcular el centro de la línea detectada
            
            # Convertir de vuelta a ROS y publicar
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.publisher.publish(ros_image)
            
            # --- FIN DE LA CORRECCIÓN ---

        except Exception as e:
            self.get_logger().error(f'Error en callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = QCarAI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
