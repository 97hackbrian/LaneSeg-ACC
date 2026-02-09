#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from qcar2_interfaces.msg import MotorCommands


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class YellowLineFollowerController(Node):
    """
    Control PID para alinear el LANE CENTER con el centro de la imagen:
      - setpoint = 0
      - error = /lane/center/error  (Float32, [-1,+1])

    Publica MotorCommands EXACTAMENTE como tu teleop:
      msg.motor_names = ['motor_throttle', 'steering_angle']
      msg.values      = [steering_angle, motor_throttle]
    """

    def __init__(self):
        super().__init__('yellow_line_follower_controller')

        # Topics
        self.declare_parameter('error_topic', '/lane/center/error')
        self.declare_parameter('visible_topic', '/lane/center/visible')
        self.declare_parameter('cmd_topic', '/qcar2_motor_speed_cmd')

        # PID
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.08)
        self.declare_parameter('integral_limit', 0.8)

        # Limits / speed
        self.declare_parameter('max_angle', 0.45)     # rad
        self.declare_parameter('base_speed', 0.35)
        self.declare_parameter('min_speed', 0.15)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('slowdown_gain', 0.65)  # baja speed con |error|

        # Safety / rate
        self.declare_parameter('lost_timeout', 0.35)
        self.declare_parameter('rate_hz', 20.0)

        self.error_topic = self.get_parameter('error_topic').value
        self.visible_topic = self.get_parameter('visible_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value

        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.integral_limit = float(self.get_parameter('integral_limit').value)

        self.max_angle = float(self.get_parameter('max_angle').value)
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.slowdown_gain = float(self.get_parameter('slowdown_gain').value)

        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        # State
        self.last_error = 0.0
        self.visible = False
        self.last_msg_time = self.get_clock().now()

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()
        self.first = True

        # Subs/Pub
        self.sub_err = self.create_subscription(Float32, self.error_topic, self.cb_error, 10)
        self.sub_vis = self.create_subscription(Bool, self.visible_topic, self.cb_visible, 10)
        self.pub_cmd = self.create_publisher(MotorCommands, self.cmd_topic, 10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        self.get_logger().info("YellowLineFollowerController (PID) started")
        self.get_logger().info(f"  error_topic  : {self.error_topic}")
        self.get_logger().info(f"  visible_topic: {self.visible_topic}")
        self.get_logger().info(f"  cmd_topic    : {self.cmd_topic}")

    def cb_error(self, msg: Float32):
        self.last_error = float(msg.data)
        self.last_msg_time = self.get_clock().now()

    def cb_visible(self, msg: Bool):
        self.visible = bool(msg.data)
        self.last_msg_time = self.get_clock().now()

    def publish_motorcommands(self, steering_angle: float, motor_throttle: float):
        msg = MotorCommands()
        # EXACTAMENTE como tu teleop
        msg.motor_names = ['motor_throttle', 'steering_angle']
        msg.values = [float(steering_angle), float(motor_throttle)]
        self.pub_cmd.publish(msg)

    def control_loop(self):
        # Safety: si no llegan msgs recientes, frena
        dt_lost = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
        if dt_lost > self.lost_timeout:
            self.publish_motorcommands(0.0, 0.0)
            self.integral = 0.0
            self.first = True
            return

        # Si aún no hay lane center válido, frena
        if not self.visible:
            self.publish_motorcommands(0.0, 0.0)
            self.integral = 0.0
            self.first = True
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds * 1e-9
        if dt <= 1e-6:
            dt = 1.0 / self.rate_hz
        self.prev_time = now

        # Error (setpoint=0)
        e = clamp(self.last_error, -1.0, 1.0)

        # PID
        if self.first:
            self.prev_error = e
            self.first = False

        self.integral += e * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (e - self.prev_error) / dt
        self.prev_error = e

        u = (self.kp * e) + (self.ki * self.integral) + (self.kd * derivative)

        # Convención: si error > 0 (lane center a la derecha), queremos girar a la derecha.
        # En muchos setups el steering positivo gira a la izquierda, por eso aplicamos signo negativo.
        steering = -u
        steering = clamp(steering, -self.max_angle, self.max_angle)

        # Velocidad adaptativa (más error => más lento)
        speed = self.base_speed * (1.0 - self.slowdown_gain * abs(e))
        speed = clamp(speed, self.min_speed, self.max_speed)

        self.publish_motorcommands(steering, speed)


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
