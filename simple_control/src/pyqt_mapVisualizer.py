#!/usr/bin/env python3
import sys, os, threading
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
    QPushButton, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItemGroup
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen
from PyQt5.QtCore import Qt, QPointF, QMetaObject, Q_ARG, pyqtSlot

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from innoagri_msgs.action import NavigateWaypoints
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose2D



class MyGraphicsView(QGraphicsView):
    """
    QGraphicsView에서 직접 마우스 이벤트를 처리해,
    - 클릭으로 경로 입력 시작/종료
    - 마우스 이동으로 선 긋기
    등의 로직을 구현.
    """
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.NoDrag)  # 드래그 모드 끔

        # 마우스 트래킹을 켜면, 마우스가 눌리지 않은 상태에서도 moveEvent가 전달됨.
        # 여기서는 '클릭 후 이동'이 주 목적이라 꼭 필요치는 않지만,
        # 원하는 경우 주석 해제 가능.
        self.setMouseTracking(True)

        # 경로 관련 속성들
        self.is_drawing = False
        self.last_waypoint = None
        self.min_distance = 10  # 마우스가 이동해야 새 웨이포인트로 인정
        self.waypoints = []
        self.path_lines = []

    def mouseDoubleClickEvent(self, event):
        # 더블클릭 이벤트를 무시
        pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            position = self.mapToScene(event.pos())
            print(f"[View] Left click at: {position}")

            if not self.is_drawing:
                # 경로 입력 시작
                self.is_drawing = True
                self.waypoints.clear()
                self.path_lines.clear()
                self.last_waypoint = position

                # 첫 웨이포인트 등록
                self.waypoints.append((position.x(), position.y()))
                self.draw_waypoint((position.x(), position.y()))
            else:
                # 경로 입력 종료
                self.is_drawing = False
                self.waypoints.append((position.x(), position.y()))
                self.draw_waypoint((position.x(), position.y()))
                self.last_waypoint = None
                print(f"[View] Path completed with waypoints: {self.waypoints}")

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.last_waypoint:
            current_position = self.mapToScene(event.pos())
            distance = (
                (current_position.x() - self.last_waypoint.x())**2
                + (current_position.y() - self.last_waypoint.y())**2
            )**0.5

            if distance >= self.min_distance:
                self.waypoints.append((current_position.x(), current_position.y()))
                line = QGraphicsLineItem(
                    self.last_waypoint.x(), self.last_waypoint.y(),
                    current_position.x(), current_position.y()
                )
                line.setPen(QPen(Qt.green, 0.5))
                # scene()은 MyGraphicsView에 연결된 QGraphicsScene을 반환
                self.scene().addItem(line)
                self.path_lines.append(line)

                self.draw_waypoint((current_position.x(), current_position.y()))
                self.last_waypoint = current_position

        super().mouseMoveEvent(event)

    def draw_waypoint(self, waypoint):
        """단순히 빨간 원을 그려주는 함수."""
        waypoint_item = QGraphicsEllipseItem(
            waypoint[0] - 1.5, waypoint[1] - 1.5, 3, 3
        )
        waypoint_item.setBrush(Qt.red)
        self.scene().addItem(waypoint_item)

    def clear_path(self):
        """씬 위에 그려진 웨이포인트와 라인을 모두 제거."""
        # 먼저 선부터 제거
        for line in self.path_lines:
            self.scene().removeItem(line)
        self.path_lines.clear()

        # 웨이포인트로 표시한 원도 제거
        # (장난스럽게 모든 QGraphicsEllipseItem을 찾아 지우는 방법)
        for item in self.scene().items():
            if isinstance(item, QGraphicsEllipseItem):
                self.scene().removeItem(item)

        self.waypoints.clear()
        self.last_waypoint = None
        self.is_drawing = False
        print("[View] Path cleared.")

    def set_waypoint_color(self, index, color=Qt.green):
        """
        index번째 웨이포인트의 색을 변경 (예: 지나간 웨이포인트를 초록색으로)
        """
        if 0 <= index < len(self.waypoint_items):
            item = self.waypoint_items[index]
            item.setBrush(color)

class OccupancyMapViewer(QMainWindow):
    def __init__(self, map_file="occupancy_map.npz"):
        super().__init__()
        self.setWindowTitle("Occupancy Map Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # Load the Occupancy Map
        self.load_occupancy_map(map_file)

        # Graphics View / Scene 설정
        self.scene = QGraphicsScene()
        self.view = MyGraphicsView(self.scene, self)
        self.view.setGeometry(10, 10, 980, 780)

        # Robot
        self.robot_position = None
        self.robot_orientation = 0
        self.robot_item = None  # 여기에서 robot_item 초기화

        # 버튼들
        self.drive_button = QPushButton("Drive", self)
        self.drive_button.setGeometry(10, 750, 100, 30)
        self.drive_button.clicked.connect(self.start_drive)

        self.clear_button = QPushButton("Clear Path", self)
        self.clear_button.setGeometry(120, 750, 100, 30)
        # 클릭 시 뷰에 있는 경로들을 지움
        self.clear_button.clicked.connect(self.view.clear_path)

        # 맵 그리기
        self.draw_map()

    def load_occupancy_map(self, map_file):
        try:
            data = np.load(map_file)
            self.map_data = data["map_data"]
            self.x_bins = data["x_bins"]
            self.y_bins = data["y_bins"]
            print(f"Loaded occupancy map with shape: {self.map_data.shape}")
        except Exception as e:
            print(f"Error loading map file: {e}")
            self.map_data = None
            self.x_bins = None
            self.y_bins = None

    def draw_map(self):
        if self.map_data is None:
            print("No occupancy map data to draw.")
            return

        height, width = self.map_data.shape
        print(f"Drawing map with size: {width}x{height}")

        qimage = QImage(height, width, QImage.Format_Grayscale8)
        for y in range(height):
            for x in range(width):
                value = 255 if self.map_data[y, x] > 0 else 0
                qimage.setPixel(height- y, x, QColor(value, value, value).rgb())

        pixmap = QPixmap.fromImage(qimage)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        if self.robot_position:
            self.draw_robot()


    def update_robot_position(self, x, y, yaw):
        """
        x,y는 occupancy_map 좌표계 (실제 부동소수).
        x_bins,y_bins를 사용해 grid index로 변환한 뒤,
        QGraphicsScene 상의 위치로 매핑해 로봇을 그린다.
        """
        if self.map_data is None or self.x_bins is None or self.y_bins is None:
            print("Map data or bins are not loaded.")
            return

        # x,y -> grid index
        x_index = np.searchsorted(self.x_bins, x) - 1
        y_index = np.searchsorted(self.y_bins, y) - 1

        # 범위 체크
        if 0 <= x_index < len(self.x_bins) and 0 <= y_index < len(self.y_bins):
            # QImage 그릴 때 row=0이 아래쪽이므로, y축 뒤집음
            # row = y_index → scene_y = (height - row - 1)
            scene_x = self.map_data.shape[0] -1 -x_index
            scene_y = y_index

            # UI 업데이트 함수
            def update_ui():
                # 이전 로봇 아이템 삭제
                if self.robot_item is not None:
                    self.scene.removeItem(self.robot_item)

                # 새 로봇 아이템: (원 + 화살표)를 묶은 그룹
                group = QGraphicsItemGroup()

                # 1) 파란 원 (로봇의 위치)
                radius = 6
                circle_item = QGraphicsEllipseItem(
                    scene_x - radius/2, scene_y - radius/2, radius, radius
                )
                circle_item.setBrush(Qt.blue)
                group.addToGroup(circle_item)

                # 2) 로봇 진행방향 표시 (화살표 혹은 선)
                # yaw=0 → x축(오른쪽) 방향으로 화살표
                # scene에서 y축은 아래로 증가하므로 sin(yaw)는 위아래가 반전.
                arrow_len = 12
                dx = arrow_len * math.cos(yaw)
                dy = arrow_len * math.sin(yaw)
                # scene상에서는 y축이 아래로 증가하므로, 실제로는 dy를 뒤집어 준다:
                line_item = QGraphicsLineItem(
                    scene_x, scene_y,
                    scene_x + dx, scene_y - dy
                )
                pen = QPen(Qt.blue, 2)
                line_item.setPen(pen)
                group.addToGroup(line_item)

                self.scene.addItem(group)
                self.robot_item = group

            # 메인 스레드에서 실행되도록 invokeMethod 사용
            QMetaObject.invokeMethod(
                self,
                "_execute_ui_update",
                Qt.QueuedConnection,
                Q_ARG(object, update_ui)
            )
        else:
            print("Robot position is out of map bounds.")

    @pyqtSlot(object)
    def _execute_ui_update(self, update_function):
        """QMetaObject.invokeMethod로 전달된 작업 실행."""
        update_function()



    def wheelEvent(self, event):
        """맵 줌 기능."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:  # Zoom in
            self.view.scale(zoom_factor, zoom_factor)
            print("Zooming in.")
        else:  # Zoom out
            self.view.scale(1 / zoom_factor, 1 / zoom_factor)
            print("Zooming out.")
                
    def scene_to_map_coords(self, sx, sy):
        """
        scene(뷰) 상의 좌표 (sx, sy)를
        occupancy map의 실제 (x, y) (예: 미터 단위)로 변환.

        - scene_x = self.map_data.shape[0] - 1 - x_index
        - scene_y = y_index
        의 역연산:
        x_index = self.map_data.shape[0] - 1 - scene_x
        y_index = scene_y

        변환 뒤 x = x_bins[x_index], y = y_bins[y_index].
        """
        # 먼저 float로 들어온 scene 좌표를 index로 환산 (정수화)
        x_index_float = self.map_data.shape[0] - 1 - sx
        y_index_float = sy

        # 적절히 반올림 혹은 버림
        x_index = int(round(x_index_float))
        y_index = int(round(y_index_float))

        # 맵 범위 체크
        if not (0 <= x_index < len(self.x_bins)) or not (0 <= y_index < len(self.y_bins)):
            print(f"[scene_to_map_coords] Out of range: x_index={x_index}, y_index={y_index}")
            # 필요한 경우 예외처리나 클램핑 로직 추가
            # 여기서는 그냥 (0,0) 등으로 반환
            return 0.0, 0.0

        # 실제 물리 좌표 (미터) 추출
        real_x = self.x_bins[x_index]
        real_y = self.y_bins[y_index]
        return real_x, real_y


    def start_drive(self):
        """
        Drive 버튼: 뷰에 있는 waypoints (scene 좌표)를
        -> Occupancy map의 실제 좌표계로 변환 -> 액션 클라이언트로 전송
        """
        waypoints_scene = self.view.waypoints  # [(sx, sy), (sx2, sy2), ...]
        print("Drive started with waypoints (scene coords):", waypoints_scene)

        # 1) 씬->맵 변환
        waypoints_map = []
        for (sx, sy) in waypoints_scene:
            rx, ry = self.scene_to_map_coords(sx, sy)
            waypoints_map.append((rx, ry))

        print("Converted to occupancy map coords:", waypoints_map)

        # 2) 액션 클라이언트에 goal 전송
        if action_client_node is not None:
            action_client_node.send_waypoints_goal(waypoints_map)
        else:
            print("[Viewer] No action client node available.")


#####################################################
# 액션 클라이언트 예시
#####################################################

class ActionClientNode(Node):
    """
    my_robot_interfaces/action/NavigateWaypoints 액션 서버에 goal을 보내고,
    feedback (current_waypoint) 을 받아서 PyQt뷰의 웨이포인트 색상 갱신
    """
    def __init__(self, viewer):
        super().__init__('waypoint_action_client_node')
        self.viewer = viewer
        self._action_client = ActionClient(self, NavigateWaypoints, 'navigate_waypoints')

    def send_waypoints_goal(self, waypoints):
        """
        waypoints: [ (x1,y1), (x2,y2), ... ]
        -> Pose2D[] 로 변환 후 액션 goal 전송
        """
        goal_msg = NavigateWaypoints.Goal()
        pose_list = []
        for (wx, wy) in waypoints:
            p = Pose2D()
            p.x = float(wx)
            p.y = float(wy)
            p.theta = 0.0  # 단순 예시
            pose_list.append(p)
        goal_msg.waypoints = pose_list

        self.get_logger().info(f"Sending goal with {len(pose_list)} waypoints.")

        # (1) 서버가 준비될 때까지 동기 대기 (최대 5초 등)
        server_ready = self._action_client.wait_for_server(timeout_sec=5.0)
        if not server_ready:
            self.get_logger().error("Action server not available after waiting.")
            return

        # (2) 서버가 준비되었으면 곧바로 Goal 전송
        self._goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._goal_future.add_done_callback(self.goal_response_callback)


    def _send_goal_request(self, goal_msg):
        self._goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """
        feedback_msg.feedback.current_waypoint 에 따라
        PyQt View의 웨이포인트 색상 변경.
        """
        current_idx = feedback_msg.feedback.current_waypoint
        self.get_logger().info(f"Feedback: Reached waypoint {current_idx}")
        # PyQt UI 갱신
        def update_ui():
            self.viewer.view.set_waypoint_color(current_idx, Qt.green)
        # 메인 스레드에서 실행
        QMetaObject.invokeMethod(
            self.viewer,
            "_execute_ui_update",
            Qt.QueuedConnection,
            Q_ARG(object, update_ui)
        )

    def result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info("Navigation completed successfully!")
        else:
            self.get_logger().info("Navigation failed or aborted.")


#####################################################
# 기존의 ROS2Subscriber + ActionClient 통합
#####################################################

class ROS2Subscriber(Node):
    def __init__(self, viewer):
        super().__init__('pyqt_localization_viewer')
        self.viewer = viewer
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/localization_2d_pose',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        if len(msg.data) < 3:
            self.get_logger().error("Invalid /localization_2d_pose message received.")
            return
        x, y, yaw = msg.data
        self.viewer.update_robot_position(x, y, yaw)


#####################################################
# 멀티 노드 스레드 실행 (구독 + 액션클라이언트)
#####################################################
action_client_node = None

def run_ros2_node(viewer):
    rclpy.init()
    # 1) 위치 구독자 노드
    subscriber_node = ROS2Subscriber(viewer)
    # 2) 액션 클라이언트 노드
    global action_client_node
    action_client_node = ActionClientNode(viewer)

    try:
        # 두 노드를 동시에 spin하기 위해 MultiThreadedExecutor 등 사용 가능
        # 여기서는 간단하게 SingleThread로 spin_some() 반복
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(subscriber_node)
        executor.add_node(action_client_node)
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS2 node...")
    finally:
        subscriber_node.destroy_node()
        action_client_node.destroy_node()
        rclpy.shutdown()


#####################################################
# 메인 함수
#####################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 맵 로드
    current_dir = os.path.dirname(os.path.realpath(__file__))
    map_file_path = os.path.join(current_dir, 'occupancy_map.npz')
    viewer = OccupancyMapViewer(map_file_path)
    viewer.show()

    # ROS2 백그라운드 스레드
    import threading
    ros2_thread = threading.Thread(target=run_ros2_node, args=(viewer,), daemon=True)
    ros2_thread.start()

    sys.exit(app.exec_())
