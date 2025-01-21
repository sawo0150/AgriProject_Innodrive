#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

# 메시지
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D

# 우리가 정의한 Action (my_robot_interfaces)
from innoagri_msgs.action import NavigateWaypoints

# stanley_controller.py 에 있는 함수/클래스 가져온다고 가정
# (예시 코드: https://github.com/... 
#   또는 질문에서 제시된 stanley 예시 코드에 맞춰 import)
from simple_control.stanley_controller import (
    State, stanley_control, pid_control,
    calc_target_index, dt, L
)

import numpy as np
import math
import threading
import time

class StanleyActionServerNode(Node):

    def __init__(self):
        super().__init__('stanley_action_server_node')
        self.get_logger().info("Stanley Action Server Node Started!")

        # 로봇 현재 상태 (2D)
        self.state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)

        # Subscribe: /localization_2d_pose
        self.pose_sub = self.create_subscription(
            Float64MultiArray,
            '/localization_2d_pose',
            self.pose_callback,
            10
        )

        # Publish: /cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Stanley 제어용 경로
        self.cx = []
        self.cy = []
        self.cyaw = []
        self.last_target_idx = 0
        self.target_speed = 1.0  # [m/s] (임의)

        # Action Server 생성
        # NavigateWaypoints.action:
        #   goal: Pose2D[] waypoints
        #   result: bool success
        #   feedback: uint32 current_waypoint
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

        # 주기적 제어를 별도 Timer로 할 수도 있지만,
        # 여기서는 Action의 실행 쓰레드 내에서 루프 돌며 Stanley 제어
        # self.control_timer = self.create_timer(dt, self.control_loop)

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

        # 2) waypoints -> (cx, cy, cyaw) 변환
        #    - 단순히 모든 점을 이어서 샘플링하거나, spline 보간
        #    - 여기선 간단히 '직선 분할' 예시
        #    - (실제로는 cubic_spline_planner.calc_spline_course() 등을 활용)
        self.cx, self.cy, self.cyaw = self.convert_waypoints_to_path(waypoints)

        # 3) 초기화
        self.is_navigating = True
        self.last_target_idx = 0
        current_waypoint_idx, _ = calc_target_index(self.state, self.cx, self.cy)

        feedback_msg = NavigateWaypoints.Feedback()
        result_msg = NavigateWaypoints.Result()

        # 4) Stanley 제어 루프
        rate_hz = 10.0
        rate_dt = 1.0 / rate_hz
        max_loop = 10000  # 안전장치 (최대 루프 횟수)
        loop_count = 0

        self.get_logger().info(f"Start navigating with {len(self.cx)} path points...")

        while rclpy.ok() and self.is_navigating:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled!")
                goal_handle.canceled()
                result_msg.success = False
                return result_msg

            # 4-1) 속도 제어 (단순 P)
            accel = pid_control(self.target_speed, self.state.v)

            # 4-2) Stanley
            delta, current_waypoint_idx = stanley_control(
                self.state, self.cx, self.cy, self.cyaw, current_waypoint_idx
            )

            # 4-3) (시뮬레이션) state 업데이트
            # 실제 로봇이라면, self.state는 /localization_2d_pose 콜백에서만 업데이트
            # -> 여기선 예시상 시뮬레이션도 겸하므로 계속 update
            self.state.update(accel, delta)

            # 4-4) /cmd_vel 퍼블리시
            twist = Twist()
            twist.linear.x = self.state.v
            twist.angular.z = (self.state.v / L) * math.tan(delta)
            self.cmd_pub.publish(twist)

            # 4-5) Feedback: 현재 '웨이포인트 인덱스' 혹은 'path 인덱스'
            #    - 여기선 path 인덱스로 피드백하되,
            #      실제로는 "현재 웨이포인트 몇 번째인가"를 구분할 수도 있음
            feedback_msg.current_waypoint = int(current_waypoint_idx)
            goal_handle.publish_feedback(feedback_msg)

            # 4-6) 종료 조건:
            #    a) current_waypoint_idx가 마지막 인덱스 근처
            #    b) 혹은 속도가 매우 낮고 목표 가까이 도달
            if current_waypoint_idx >= (len(self.cx) - 2):
                self.get_logger().info("Reached final path point!")
                result_msg.success = True
                goal_handle.succeed()
                break

            time.sleep(rate_dt)
            loop_count += 1
            if loop_count > max_loop:
                self.get_logger().warn("Loop exceeded max_loop => abort")
                result_msg.success = False
                goal_handle.abort()
                break

        # 주행 끝
        self.is_navigating = False
        # 정지 cmd_vel
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)

        return result_msg

    # -----------------------------
    # 콜백 & 기타 함수
    # -----------------------------
    def pose_callback(self, msg):
        """
        /localization_2d_pose = Float64MultiArray(data=[x, y, yaw])
        이 콜백에서 실제 로봇 상태를 업데이트
        (시뮬/개념적으로는 stanley_control() 후 update() 쓰지만,
         실제 하드웨어는 바퀴 동작 결과가 다시 오돔/Localization으로 들어옴)
        """
        if len(msg.data) >= 3:
            x, y, yaw = msg.data[:3]
            self.state.x = x
            self.state.y = y
            self.state.yaw = yaw
            # self.state.v는 직접 속도 추정하거나 별도 토픽 구독 가능
            # 여기서는 간단히 유지 (또는 0)

    def convert_waypoints_to_path(self, waypoints):
        """
        waypoints: list of Pose2D
        return: cx, cy, cyaw
          - stanley_control 함수가 요구하는 경로
        여기서는 각 웨이포인트 사이를 직선 분할(간단) + yaw 추정
        """
        cx = []
        cy = []
        cyaw = []

        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i].x, waypoints[i].y
            x1, y1 = waypoints[i+1].x, waypoints[i+1].y

            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            steps = max(int(dist / 0.1), 1)  # 0.1m 간격으로 샘플링
            for s in range(steps):
                r = float(s) / steps
                xx = x0 + dx * r
                yy = y0 + dy * r
                cx.append(xx)
                cy.append(yy)

            # 마지막점은 i+1번째 웨이포인트
            cx.append(x1)
            cy.append(y1)

        # yaw(방향) 계산
        for i in range(len(cx)-1):
            dx = cx[i+1] - cx[i]
            dy = cy[i+1] - cy[i]
            heading = math.atan2(dy, dx)
            cyaw.append(heading)
        # 마지막은 직전과 동일
        if len(cx) > 1:
            cyaw.append(cyaw[-1])
        else:
            cyaw.append(0.0)

        return cx, cy, cyaw

def main(args=None):
    rclpy.init(args=args)
    node = StanleyActionServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt => shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
