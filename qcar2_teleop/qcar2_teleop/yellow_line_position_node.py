#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class YellowLinePositionNode(Node):
    """
    Lee la máscara BGR (/segmentation/color_mask) y:
      - Detecta amarillo (lane) por HSV
      - Detecta borde derecho (edge) como ROJO (0,0,255) en la máscara
      - Predice lane_center_x:
          * Si hay yellow y edge: midpoint + aprende lane_width_px
          * Si hay edge pero no yellow: lane_center = edge - 0.5*lane_width_px
          * Si NO hay edge: HOLD infinito del lane center previo

    Publica:
      - /lane/yellow/error   (Float32)
      - /lane/yellow/visible (Bool)
      - /lane/center/error   (Float32)
      - /lane/center/visible (Bool)
    """

    def __init__(self):
        super().__init__('yellow_line_position_node')

        # Topics
        self.declare_parameter('mask_topic', '/segmentation/color_mask')
        self.declare_parameter('yellow_error_topic', '/lane/yellow/error')
        self.declare_parameter('yellow_visible_topic', '/lane/yellow/visible')
        self.declare_parameter('center_error_topic', '/lane/center/error')
        self.declare_parameter('center_visible_topic', '/lane/center/visible')

        # ROI
        self.declare_parameter('use_bottom_ratio', 0.45)

        # Thresholds
        self.declare_parameter('min_yellow_pixels', 120)
        self.declare_parameter('min_edge_pixels', 180)

        # NEW: sanity check edge (si el borde aparece demasiado a la izquierda, es falso)
        self.declare_parameter('min_edge_x_ratio', 0.62)

        # Debug window
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'lane_center_prediction')

        # Lane width learning
        self.declare_parameter('default_lane_width_ratio', 0.40)
        # <-- CALIBRA ESTE VALOR: 0.45~0.70 según tu mapa/cámara.
        #     Si el carro se va muy cerca de la acera, SUBE el ratio.
        #     Si el carro se va muy al centro/izquierda, BAJA el ratio.

        self.declare_parameter('lane_width_ema_alpha', 0.20)
        self.declare_parameter('min_lane_width_ratio', 0.25)
        self.declare_parameter('max_lane_width_ratio', 0.95)

        self.mask_topic = self.get_parameter('mask_topic').value
        self.yellow_error_topic = self.get_parameter('yellow_error_topic').value
        self.yellow_visible_topic = self.get_parameter('yellow_visible_topic').value
        self.center_error_topic = self.get_parameter('center_error_topic').value
        self.center_visible_topic = self.get_parameter('center_visible_topic').value

        self.use_bottom_ratio = float(self.get_parameter('use_bottom_ratio').value)
        self.min_yellow_pixels = int(self.get_parameter('min_yellow_pixels').value)
        self.min_edge_pixels = int(self.get_parameter('min_edge_pixels').value)
        self.min_edge_x_ratio = float(self.get_parameter('min_edge_x_ratio').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = self.get_parameter('window_name').value

        self.default_lane_width_ratio = float(self.get_parameter('default_lane_width_ratio').value)
        self.lane_width_ema_alpha = float(self.get_parameter('lane_width_ema_alpha').value)
        self.min_lane_width_ratio = float(self.get_parameter('min_lane_width_ratio').value)
        self.max_lane_width_ratio = float(self.get_parameter('max_lane_width_ratio').value)

        self.bridge = CvBridge()

        # Hold infinito
        self.last_lane_center_x = None
        self.center_has_value = False

        # Lane width memory
        self.lane_width_px = None

        self.sub = self.create_subscription(Image, self.mask_topic, self.cb_mask, 10)
        self.pub_y_err = self.create_publisher(Float32, self.yellow_error_topic, 10)
        self.pub_y_vis = self.create_publisher(Bool, self.yellow_visible_topic, 10)
        self.pub_c_err = self.create_publisher(Float32, self.center_error_topic, 10)
        self.pub_c_vis = self.create_publisher(Bool, self.center_visible_topic, 10)

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("YellowLinePositionNode started (robust edge + lane width learning)")
        self.get_logger().info(f"  mask_topic: {self.mask_topic}")
        self.get_logger().info(f"  center_error_topic: {self.center_error_topic}")
        self.get_logger().info(f"  default_lane_width_ratio: {self.default_lane_width_ratio}")
        self.get_logger().info(f"  min_edge_x_ratio: {self.min_edge_x_ratio}")

    def _edge_cx_rightmost_median(self, edge_bin: np.ndarray):
        """
        Edge robusto:
        - Para cada fila y, toma el pixel edge más a la derecha.
        - Luego usa la mediana de esos rightmost_x.
        Esto ignora manchas rojas dentro de la carretera.
        """
        ys, xs = np.where(edge_bin > 0)
        if xs.size == 0:
            return None

        rightmost = {}
        for y, x in zip(ys, xs):
            prev = rightmost.get(int(y), None)
            if prev is None or x > prev:
                rightmost[int(y)] = int(x)

        rightmost_xs = np.array(list(rightmost.values()), dtype=np.float32)
        if rightmost_xs.size < 10:
            # muy pocos datos -> poco confiable
            return None

        return int(np.median(rightmost_xs))

    def cb_mask(self, msg: Image):
        mask_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if mask_bgr is None or mask_bgr.size == 0:
            return

        H, W = mask_bgr.shape[:2]
        y0 = int(H * (1.0 - self.use_bottom_ratio))
        roi = mask_bgr[y0:, :]
        h, w = roi.shape[:2]
        center_x = w // 2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 1) Amarillo (HSV)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_y = np.array([20, 120, 120], dtype=np.uint8)
        upper_y = np.array([40, 255, 255], dtype=np.uint8)
        yellow_bin = cv2.inRange(roi_hsv, lower_y, upper_y)
        yellow_bin = cv2.morphologyEx(yellow_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        yellow_bin = cv2.morphologyEx(yellow_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

        yellow_pixels = int(cv2.countNonZero(yellow_bin))
        yellow_visible = yellow_pixels >= self.min_yellow_pixels

        yellow_cx = None
        if yellow_visible:
            M = cv2.moments(yellow_bin)
            if M["m00"] > 0:
                yellow_cx = int(M["m10"] / M["m00"])
            else:
                yellow_visible = False

        if yellow_visible and yellow_cx is not None:
            yellow_error = (yellow_cx - (w / 2.0)) / (w / 2.0)
            yellow_error = float(clamp(yellow_error, -1.0, 1.0))
        else:
            yellow_error = 0.0

        self.pub_y_vis.publish(Bool(data=yellow_visible))
        self.pub_y_err.publish(Float32(data=yellow_error))

        # 2) Edge (ROJO en máscara)
        b, g, r = cv2.split(roi)
        edge_bin = ((r > 200) & (g < 80) & (b < 80)).astype(np.uint8) * 255
        edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

        edge_pixels = int(cv2.countNonZero(edge_bin))
        edge_visible = edge_pixels >= self.min_edge_pixels

        edge_cx = None
        if edge_visible:
            # ✅ REEMPLAZO CRÍTICO: ya no centroide; ahora rightmost+mediana
            edge_cx = self._edge_cx_rightmost_median(edge_bin)
            if edge_cx is None:
                edge_visible = False

        # ✅ Sanity check: si el “edge” cae muy a la izquierda, lo descartamos
        if edge_visible and edge_cx is not None:
            if edge_cx < int(self.min_edge_x_ratio * w):
                edge_visible = False
                edge_cx = None

        # 2.5) aprender lane_width_px cuando hay YELLOW + EDGE
        if edge_visible and (edge_cx is not None) and yellow_visible and (yellow_cx is not None):
            width_px_meas = float(edge_cx - yellow_cx)

            min_w = self.min_lane_width_ratio * w
            max_w = self.max_lane_width_ratio * w

            if min_w <= width_px_meas <= max_w:
                if self.lane_width_px is None:
                    self.lane_width_px = width_px_meas
                else:
                    a = float(clamp(self.lane_width_ema_alpha, 0.0, 1.0))
                    self.lane_width_px = (1.0 - a) * self.lane_width_px + a * width_px_meas

        # Si todavía no tenemos lane_width_px, usamos default
        if self.lane_width_px is None:
            self.lane_width_px = float(clamp(self.default_lane_width_ratio, 0.05, 0.99)) * w  # <-- CALIBRA default_lane_width_ratio arriba

        # 3) Lane center prediction (HOLD infinito)
        lane_center_x = None

        if edge_visible and edge_cx is not None:
            if yellow_visible and yellow_cx is not None:
                lane_center_x = int((yellow_cx + edge_cx) / 2.0)
            else:
                lane_center_x = int(edge_cx - 0.5 * float(self.lane_width_px))

            lane_center_x = int(clamp(lane_center_x, 0, w - 1))
            self.last_lane_center_x = lane_center_x
            self.center_has_value = True
        else:
            if self.last_lane_center_x is not None:
                lane_center_x = self.last_lane_center_x

        # 4) Error del center
        if lane_center_x is not None:
            center_error = (lane_center_x - (w / 2.0)) / (w / 2.0)
            center_error = float(clamp(center_error, -1.0, 1.0))
        else:
            center_error = 0.0

        self.pub_c_vis.publish(Bool(data=self.center_has_value))
        self.pub_c_err.publish(Float32(data=center_error))

        # 5) Debug window
        if self.show_window:
            dbg = roi.copy()

            cv2.line(dbg, (center_x, 0), (center_x, h), (255, 255, 255), 2)

            if yellow_cx is not None:
                cv2.circle(dbg, (yellow_cx, int(h * 0.60)), 8, (0, 255, 255), -1)
                cv2.putText(dbg, "YELLOW", (yellow_cx + 10, int(h * 0.60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if edge_cx is not None:
                cv2.circle(dbg, (edge_cx, int(h * 0.72)), 10, (0, 255, 0), -1)
                cv2.circle(dbg, (edge_cx, int(h * 0.72)), 14, (0, 0, 0), 2)
                cv2.putText(dbg, "EDGE (rightmost median)", (edge_cx + 10, int(h * 0.72)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if lane_center_x is not None:
                cv2.line(dbg, (lane_center_x, 0), (lane_center_x, h), (255, 0, 255), 2)
                cv2.circle(dbg, (lane_center_x, int(h * 0.85)), 8, (255, 0, 255), -1)

            cv2.putText(
                dbg,
                f"lane_width_px={float(self.lane_width_px):.1f} (ratio={float(self.lane_width_px)/float(w):.2f})",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                dbg,
                f"Yvis={yellow_visible} pix={yellow_pixels} | Evis={edge_visible} pix={edge_pixels} | "
                f"Yerr={yellow_error:+.2f} | Cerr={center_error:+.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(self.window_name, dbg)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.get_logger().warn("ESC: cerrando ventana debug")
                cv2.destroyWindow(self.window_name)
                self.show_window = False


def main(args=None):
    rclpy.init(args=args)
    node = YellowLinePositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
