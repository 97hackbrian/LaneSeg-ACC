#!/usr/bin/env python3
"""
QCar2 Fast Teleop — controlled by PS4 D-Pad via /ps4_dpad_cmd topic
───────────────────────────────────────────────────────────────────────
Subscribes to /ps4_dpad_cmd (geometry_msgs/Twist) published by
ps4_teleop_node (D-Pad arrows from PS4 controller).

D-Pad mapping:
  ↑  → increase speed   (+0.1)
  ↓  → decrease speed   (-0.1)
  ←  → steer left       (+0.05 rad)
  →  → steer right      (-0.05 rad)
  L1             → reset ALL to zero (speed + angle)
  △ (Triangle) → reset angle to 0
Also supports keyboard fallback:
  W/S = speed, A/D = steer, Space = brake, Ctrl+C = exit
"""

import rclpy
from rclpy.node import Node
from qcar2_interfaces.msg import MotorCommands
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

settings = termios.tcgetattr(sys.stdin)


class QCar2Teleop(Node):
    def __init__(self):
        super().__init__('qcar2_manual_teleop')
        self.pub = self.create_publisher(
            MotorCommands, '/qcar2_motor_speed_cmd', 10
        )
        self.speed = 0.0
        self.angle = 0.0

        # ── Subscribe to PS4 D-pad commands ──────────────────────────
        self.create_subscription(
            Twist, '/ps4_dpad_cmd', self._dpad_cb, 10
        )

        # ── Timer to also allow keyboard input and publish commands ──
        self.timer = self.create_timer(0.1, self._tick)

        # Debounce: only act on rising edge of D-pad
        self._prev_up = False
        self._prev_down = False
        self._prev_left = False
        self._prev_right = False

        self.get_logger().info(
            "🎮 QCar2 Teleop listo — usa D-Pad del PS4 o teclado W/S/A/D"
        )

    # ─── PS4 D-Pad callback ─────────────────────────────────────────
    def _dpad_cb(self, msg: Twist):
        up    = msg.linear.x > 0.5
        down  = msg.linear.x < -0.5
        left  = msg.angular.z > 0.5
        right = msg.angular.z < -0.5
        brake = msg.linear.y > 0.5
        reset_all = msg.linear.z > 0.5
        reset_angle = msg.angular.x > 0.5

        # L1 = reset all to zero
        if reset_all:
            self.speed = 0.0
            self.angle = 0.0

        # Brake (× button)
        if brake:
            self.speed = 0.0
            self.angle = 0.0

        # Reset angle (△ button)
        if reset_angle:
            self.angle = 0.0

        # Rising edge detection for D-pad (so holding doesn't spam)
        if up and not self._prev_up:
            self.speed += 0.1
        if down and not self._prev_down:
            self.speed -= 0.1
        if left and not self._prev_left:
            self.angle += 0.05
        if right and not self._prev_right:
            self.angle -= 0.05

        self._prev_up = up
        self._prev_down = down
        self._prev_left = left
        self._prev_right = right

    # ─── Timer tick: keyboard fallback + publish ────────────────────
    def _tick(self):
        # Keyboard fallback
        key = self._getKey()
        if key == 'w':
            self.speed += 0.1
        elif key == 's':
            self.speed -= 0.1
        elif key == 'a':
            self.angle += 0.05
        elif key == 'd':
            self.angle -= 0.05
        elif key == ' ':
            self.speed = 0.0
            self.angle = 0.0
        elif key == '\x03':
            rclpy.shutdown()
            return

        # Clamp steering angle to ±0.6 rad
        self.angle = max(-0.6, min(0.6, self.angle))
        self.speed = max(-0.7, min(0.7, self.speed))


        # Publish motor commands
        msg = MotorCommands()
        msg.motor_names = ['motor_throttle', 'steering_angle']
        msg.values = [float(self.angle), float(self.speed)]
        self.pub.publish(msg)

        sys.stdout.write(
            f"\rAng: {self.angle:.2f} m/s | Vel: {self.speed:.2f} rad   "
        )
        sys.stdout.flush()

    def _getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key


def main():
    rclpy.init()
    node = QCar2Teleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the car
        msg = MotorCommands()
        msg.motor_names = ['motor_throttle', 'steering_angle']
        msg.values = [0.0, 0.0]
        node.pub.publish(msg)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.get_logger().info("🛑 Teleop detenido — QCar2 parado")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
