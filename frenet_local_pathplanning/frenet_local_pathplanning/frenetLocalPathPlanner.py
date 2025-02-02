#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

import cv2
import concurrent.futures

# 메시지
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image, CameraInfo

# 우리가 정의한 Action (my_robot_interfaces)
from innoagri_msgs.action import NavigateWaypoints


import numpy as np
import math
import threading
import time

import frenet_local_pathplanning.include.frenet_optimal_trajectory as fot
from frenet_local_pathplanning.include.cubic_spline_planner import CubicSpline2D
from frenet_local_pathplanning.include.cartesian_frenet_converter import CartesianFrenetConverter
from frenet_local_pathplanning.include.groundedSAM2 import GroundedSAM2
from frenet_local_pathplanning.include.waypointVisualizer import WaypointVisualizer
from frenet_local_pathplanning.include.localization_converter import localization_converter

MAX_SPEED = 1.3  # maximum speed [m/s]
MAX_ACCEL = 5.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]

window_size = 5


# Initialize the CV bridge
bridge = CvBridge()

class frenetLocalPathPlanner(Node):

    def __init__(self):
        super().__init__('frenetLocalPathPlanner')
        self.get_logger().info("Stanley Action Server Node Started!")

        # 로봇 현재 상태 (2D)
        self.current_pose = Pose2D(x=0.0, y=0.0, theta=0.0)

        self.subscription = self.create_subscription(
            Image,
            '/sim_camera/rgb',
            self.image_callback,
            2
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/sim_camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.localization_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/rtabmap/localization_pose',
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
        self.linear_speed = 0
        self.angular_speed = 0


        # 주행 중인지 여부
        self.is_navigating = False
        self.navigation_thread = None

        #frenet local path planning 관련 init
        self.cubicSplineWay = None
        self.tx = None
        self.ty = None
        self.tyaw = None
        self.tk = None
        self.rdk = None
        self.ts = None
        self.localPP_waypoints = None

        self.localizationConverter = localization_converter()

        self.SAM2 = GroundedSAM2()
        self.cv_image = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.latest_sam2_future = None

        self.waypointVisualizer = WaypointVisualizer()



    def image_callback(self, msg):
        # Convert ROS2 Image to OpenCV format
        self.cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # numpy.ndarray

        # annotated_frame, ground_mask = self.SAM2.image_callback(self.cv_image)
        # cv2.imshow("annotated_frame", annotated_frame)
        # cv2.waitKey(1)

    def camera_info_callback(self, msg: CameraInfo):
        self.waypointVisualizer.camera_info_callback(msg)

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
        
        waypointX = [wp.x for wp in waypoints]
        waypointY = [wp.y for wp in waypoints]

        self.tx, self.ty, self.tyaw, self.tk, self.rdk, self.ts, self.cubicSplineWay = fot.generate_target_course(waypointX, waypointY)

        self.get_logger().info(f"DrivingDistance : {self.cubicSplineWay.s[-1]}")

        
        self.is_navigating = True
        feedback_msg = NavigateWaypoints.Feedback()
        result_msg = NavigateWaypoints.Result()

        # 경로를 따라가며 루프 실행
        current_waypoint_idx = 0
        rate_hz = 5.0
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
            
            if self.localPP_waypoints is None:
                continue
            target_waypoint = (self.localPP_waypoints.x[1], self.localPP_waypoints.y[1])
            # del waypoint_copy[:current_waypoint_idx]
            # 목표 방향 계산
            dx = target_waypoint[0] - self.current_pose.x
            dy = target_waypoint[1] - self.current_pose.y
            target_angle = math.atan2(dy, dx)

            # 현재 방향과 목표 방향의 차이 계산
            # print(target_angle, self.current_pose.theta)
            angle_diff = self.normalize_angle(target_angle - self.current_pose.theta)

            # 선속도와 각속도 설정
            self.linear_speed = 1.0  # m/s
            self.angular_speed = 2.0 * angle_diff * self.linear_speed  # P 제어기로 각속도 계산

            # print(linear_speed, angular_speed, dx, dy)
            # cmd_vel 메시지 퍼블리시
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = max(min(self.angular_speed, 1.0), -1.0)
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
        localization_pose2D = self.localizationConverter.localization_pose_callback(msg)
        
        self.current_pose.x = localization_pose2D[0]
        self.current_pose.y = localization_pose2D[1]
        self.current_pose.theta = localization_pose2D[2]+math.pi

        if self.is_navigating:
            # 이전 SAM2 작업이 완료되었거나 없으면 새 작업 실행
            if self.latest_sam2_future is None or self.latest_sam2_future.done():
                self.latest_sam2_future = self.executor.submit(self.SAM2.image_callback, self.cv_image)
        
            # SAM2 연산 결과가 준비되었으면 후속 처리 진행
            if self.latest_sam2_future is not None and self.latest_sam2_future.done():
                try:
                    annotated_frame, ground_mask = self.latest_sam2_future.result()
                    cv2.imshow("annotated_frame", annotated_frame)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().error(f"SAM2 processing failed: {e}")

            self.waypointVisualizer.localization_pose_callback(msg)
            
            s_coor, d_coor = self.convert_to_frenet(self.current_pose.x,
                                                    self.current_pose.y,
                                                    self.current_pose.theta,
                                                    self.linear_speed,
                                                    self.angular_speed,
                                                    self.tx,
                                                    self.ty,
                                                    self.tyaw,
                                                    self.tk,
                                                    self.rdk,
                                                    self.ts,
                                                    self.cubicSplineWay)
            [path, fpdict] = self.frenet_optimal_planning(
                self.cubicSplineWay, s_coor[0], s_coor[1], s_coor[2], d_coor[0], d_coor[1], d_coor[2], ground_mask)
            self.localPP_waypoints = path

    def convert_to_frenet(self, x, y, theta, v, omega, cx, cy, cyaw, ck, cdk, cs, csp):
        """
        SLAM으로 받은 (x,y) 좌표를 Frenet 좌표계 (s,d)로 변환합니다.
        
        Parameters:
            x, y   : 현재 차량의 월드 좌표
            cx, cy : 기준 경로의 x, y 좌표 배열 (충분히 촘촘하게 샘플링된)
            cyaw   : 기준 경로의 각도(heading) 배열 (라디안 단위)
            cs     : 기준 경로의 누적 거리(arc length) 배열
            
        Returns:
            s : 기준 경로 상에서의 arc length 좌표
            d : 기준 경로로부터의 수직 오프셋 (부호에 따라 좌/우 결정)
        """
        kappa = omega / (v+1e-6)

        # 1. 기준 경로상의 모든 점과의 유클리드 거리를 계산
        dx = np.array(cx) - x
        dy = np.array(cy) - y
        dists = np.hypot(dx, dy)
        
        # 2. 가장 가까운 점의 인덱스를 찾음
        min_index = np.argmin(dists)
        
        # 3. 최근접점의 정보
        rx = cx[min_index]
        ry = cy[min_index]
        rtheta = cyaw[min_index]
        rs = cs[min_index]  # 이 값이 차량의 s 좌표
        rkappa = ck[min_index]  
        rdkappa = cdk[min_index]
        a=0

        s_coor, d_coor = CartesianFrenetConverter.cartesian_to_frenet(rs, 
                                                                      rx, 
                                                                      ry, 
                                                                      rtheta, 
                                                                      rkappa, 
                                                                      rdkappa, 
                                                                      x, 
                                                                      y, 
                                                                      v, 
                                                                      a, 
                                                                      theta, 
                                                                      kappa)

        return s_coor, d_coor



    def frenet_optimal_planning(self, csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, ground_mask):
        fplist = fot.calc_frenet_paths(c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, s0)
        fplist = fot.calc_global_paths(fplist, csp)
        fpdict = self.check_paths(fplist, ground_mask)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fpdict["ok"]:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        return [best_path, fpdict]
    
    def check_paths(self, fplist, ground_mask):
        path_dict = {
            "max_speed_error": [],
            "max_accel_error": [],
            "max_curvature_error": [],
            "collision_error": [],
            "ok": [],
        }
        for i, _ in enumerate(fplist):
            if any([v > MAX_SPEED for v in fplist[i].v]):  # Max speed check
                path_dict["max_speed_error"].append(fplist[i])
            elif any([abs(a) > MAX_ACCEL for a in fplist[i].a]):  # Max accel check
                path_dict["max_accel_error"].append(fplist[i])
            elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
                path_dict["max_curvature_error"].append(fplist[i])
            elif not self.check_collision(fplist[i], ground_mask):
                path_dict["collision_error"].append(fplist[i])
            else:
                path_dict["ok"].append(fplist[i])
        return path_dict
    
    def check_collision(self, fp, ground_mask):
        if ground_mask is None:
            return True
            
        projected_waypoints = self.waypointVisualizer.cal_projectedWaypoints(fp.x, fp.y)
        
        # 모든 투영된 웨이포인트에 대해 검사
        for waypoint in projected_waypoints:
            # waypoint[1]은 평면에 투영된 점 (x,y)
            if waypoint[1] is None:
                continue
                
            x, y = int(waypoint[1][0]), int(waypoint[1][1])
            
            # 이미지 범위 체크
            if x < 0 or y < 0 or x >= ground_mask.shape[1] or y >= ground_mask.shape[0]:
                continue
                
            # 주변 영역 검사 (5x5 윈도우)
            
            half_size = window_size // 2
            
            x_start = max(0, x - half_size)
            x_end = min(ground_mask.shape[1], x + half_size + 1)
            y_start = max(0, y - half_size)
            y_end = min(ground_mask.shape[0], y + half_size + 1)
            
            # 윈도우 내의 모든 픽셀이 True(주행 가능)인지 확인
            window = ground_mask[y_start:y_end, x_start:x_end]
            if not np.all(window):
                return False
                
        return True
    
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
    node = frenetLocalPathPlanner()

    # 멀티스레드 Executor 생성
    # num_threads=2 이상으로 스레드 풀을 마련
    executor = MultiThreadedExecutor(num_threads=6)
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