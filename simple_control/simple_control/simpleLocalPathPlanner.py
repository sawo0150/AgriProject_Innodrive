#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
# 메시지
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D

# 우리가 정의한 Action (my_robot_interfaces)
from innoagri_msgs.action import NavigateWaypoints


import numpy as np
import math
import threading
import time

class SimpleWaypointFollowerNode(Node):

    def __init__(self):
        super().__init__('stanley_action_server_node')
        self.get_logger().info("Stanley Action Server Node Started!")

        # 로봇 현재 상태 (2D)
        self.current_pose = Pose2D(x=0.0, y=0.0, theta=0.0)

        # Subscribe: /localization_2d_pose
        self.pose_sub = self.create_subscription(
            Float64MultiArray,
            '/localization_2d_pose',
            self.pose_callback,
            10
        )

        # Publish: /cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._action_server = ActionServer(
            self,
            NavigateWaypoints,
            'navigate_waypoints',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # 주행 중인지 여부
        self.is_navigating = False
        self.navigation_thread = None

    # -----------------------------
    # Action Server 관련 콜백
    # -----------------------------
    def goal_callback(self, goal_request):
        """
        goal_request: NavigateWaypoints.Goal
        """
        # 여기서 goal 수락 여부를 결정
        self.get_logger().info("Received goal request!")
        # 간단히 무조건 accept
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Cancel 요청 처리
        """
        self.get_logger().info("Received cancel request!")
        # 간단히 무조건 accept
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        실제 Goal 수락 후 실행 로직
        여기서 Stanley 루프를 돌면서,
        - 피드백(current_waypoint) 발행
        - 경로 완료시 result(success) 반환
        """
        self.get_logger().info("Executing goal...")

        # 1) goal에서 waypoints (Pose2D[]) 꺼내기
        waypoints = goal_handle.request.waypoints  # list of Pose2D
        if len(waypoints) < 2:
            self.get_logger().warn("Waypoints are too few. Aborting...")
            goal_handle.abort()
            return NavigateWaypoints.Result(success=False)
        
        self.is_navigating = True
        feedback_msg = NavigateWaypoints.Feedback()
        result_msg = NavigateWaypoints.Result()

        # 경로를 따라가며 루프 실행
        current_waypoint_idx = 0
        rate_hz = 10.0
        rate_dt = 1.0 / rate_hz
        waypoint_copy = waypoints
        while rclpy.ok() and self.is_navigating:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled!")
                goal_handle.canceled()
                result_msg.success = False
                return result_msg

            # 현재 위치에서 가장 가까운 웨이포인트 찾기
            current_waypoint_idx = self.find_closest_waypoint(self.current_pose, waypoint_copy)

            # 목표 웨이포인트 설정 (현재 웨이포인트의 다음 지점)
            if current_waypoint_idx >= len(waypoint_copy) - 1:
                self.get_logger().info("Reached final waypoint!")
                result_msg.success = True
                goal_handle.succeed()
                break

            target_waypoint = waypoint_copy[current_waypoint_idx + 1]
            # del waypoint_copy[:current_waypoint_idx]
            # 목표 방향 계산
            dx = target_waypoint.x - self.current_pose.x
            dy = target_waypoint.y - self.current_pose.y
            target_angle = math.atan2(dy, dx)

            # 현재 방향과 목표 방향의 차이 계산
            print(target_angle, self.current_pose.theta)
            angle_diff = self.normalize_angle(target_angle - self.current_pose.theta)

            # 선속도와 각속도 설정
            linear_speed = 1.0  # m/s
            angular_speed = 2.0 * angle_diff * linear_speed  # P 제어기로 각속도 계산

            # print(linear_speed, angular_speed, dx, dy)
            # cmd_vel 메시지 퍼블리시
            twist = Twist()
            twist.linear.x = linear_speed
            twist.angular.z = max(min(angular_speed, 1.0), -1.0)
            self.cmd_pub.publish(twist)

            # 피드백 발행
            feedback_msg.current_waypoint = int(current_waypoint_idx)
            goal_handle.publish_feedback(feedback_msg)

            # 일정 주기로 루프 실행
            time.sleep(rate_dt)

        # 주행 종료
        self.is_navigating = False
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)

        return result_msg


    # -----------------------------
    # 콜백 & 기타 함수
    # -----------------------------
    def pose_callback(self, msg):
        """
        /localization_2d_pose = Float64MultiArray(data=[x, y, yaw])
        """
        if len(msg.data) >= 3:
            self.current_pose.x = msg.data[0]
            self.current_pose.y = msg.data[1]
            self.current_pose.theta = msg.data[2]+math.pi

    def find_closest_waypoint(self, pose, waypoints):
        """
        현재 위치에서 가장 가까운 웨이포인트 인덱스를 반환
        """
        distances = [math.hypot(wp.x - pose.x, wp.y - pose.y) for wp in waypoints]
        # print(int(np.argmin(distances)))
        return int(np.argmin(distances))

    @staticmethod
    def normalize_angle(angle):
        """
        각도를 -pi ~ pi 사이로 정규화
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = SimpleWaypointFollowerNode()

    # 멀티스레드 Executor 생성
    # num_threads=2 이상으로 스레드 풀을 마련
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()  # 여러 콜백이 동시에 실행 가능
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt => shutting down.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()