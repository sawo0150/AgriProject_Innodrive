#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Initialize pygame for keyboard input
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption('Keyboard Teleop')

        self.get_logger().info("Keyboard Teleop Initialized. Use 'WASD' keys to control. Press 'Q' to quit.")

        # Create a timer to publish at 10Hz
        self.timer = self.create_timer(0.1, self.publish_velocity)

    def get_key_states(self):
        keys = pygame.key.get_pressed()
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Map keys to velocities
        if keys[pygame.K_w]:
            self.linear_velocity = 1.0
        if keys[pygame.K_s]:
            self.linear_velocity = -1.0
        if keys[pygame.K_a]:
            self.angular_velocity = 1.0
        if keys[pygame.K_d]:
            self.angular_velocity = -1.0

    def publish_velocity(self):
        # Update key states
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.get_logger().info("Exiting...")
                rclpy.shutdown()
                pygame.quit()
                return

        self.get_key_states()

        # Publish Twist message
        twist = Twist()
        twist.linear.x = self.linear_velocity
        twist.angular.z = self.angular_velocity
        self.publisher_.publish(twist)
        self.get_logger().info(f"Published: linear={self.linear_velocity:.2f}, angular={self.angular_velocity:.2f}")

def main(args=None):
    rclpy.init(args=args)
    teleop_node = KeyboardTeleop()

    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info('Keyboard Interrupt. Shutting down.')
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
