#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from rosgraph_msgs.msg import Clock
import copy, math

from sensor_msgs_py import point_cloud2 as pc2

PI = 3.141592
class PCLTimestampSync(Node):
    def __init__(self):
        super().__init__('pcl_timestamp_sync')
        # 파라미터로 토픽 이름을 지정할 수 있습니다.
        self.declare_parameter('pcl_input_topic', '/sim_camera/depth_pcl')
        self.declare_parameter('pcl_output_topic', '/sim_camera/depth_pcl_synched')
        self.declare_parameter('clock_topic', '/clock')
        self.declare_parameter('laserscan_topic', '/sim_camera/depth_laserscan')

        pcl_input_topic = self.get_parameter('pcl_input_topic').get_parameter_value().string_value
        pcl_output_topic = self.get_parameter('pcl_output_topic').get_parameter_value().string_value
        clock_topic = self.get_parameter('clock_topic').get_parameter_value().string_value
        laserscan_topic = self.get_parameter('laserscan_topic').get_parameter_value().string_value

        self.declare_parameter('laser_scan_topic', '/scan')
        self.declare_parameter('sensor_frame', 'RSD455')  # 센서 프레임 이름
        self.declare_parameter('scan_angle_min', -PI/2+PI)
        self.declare_parameter('scan_angle_max', PI/2+PI)
        self.declare_parameter('scan_angle_increment', 0.0175)  # 약 1도
        self.declare_parameter('scan_range_min', 0.0)
        self.declare_parameter('scan_range_max', 10.0)

        # 높이 필터 관련 파라미터
        self.declare_parameter('base_link_height', 0.2)
        self.declare_parameter('rsd455_height', 0.5)
        self.declare_parameter('height_tolerance', 0.1)  # base_link와 RSD455 모두에 동일 tolerance 적용
        
        # 파라미터 값 가져오기
        self.sensor_frame = self.get_parameter('sensor_frame').get_parameter_value().string_value

        self.scan_angle_min = self.get_parameter('scan_angle_min').get_parameter_value().double_value
        self.scan_angle_max = self.get_parameter('scan_angle_max').get_parameter_value().double_value
        self.scan_angle_increment = self.get_parameter('scan_angle_increment').get_parameter_value().double_value
        self.scan_range_min = self.get_parameter('scan_range_min').get_parameter_value().double_value
        self.scan_range_max = self.get_parameter('scan_range_max').get_parameter_value

        # /clock 토픽 구독 (Clock 메시지는 msg.clock 필드에 builtin_interfaces/Time 값을 가집니다)
        self.subscription_clock = self.create_subscription(
            Clock,
            clock_topic,
            self.clock_callback,
            10
        )

        # pcl 토픽 구독 (sensor_msgs/PointCloud2)
        self.subscription_pcl = self.create_subscription(
            PointCloud2,
            pcl_input_topic,
            self.pcl_callback,
            10
        )

        # 수정된 pcl 메시지를 발행할 퍼블리셔
        self.publisher_pcl = self.create_publisher(PointCloud2, pcl_output_topic, 10)
        # LaserScan 메시지 발행 퍼블리셔
        self.publisher_laserscan = self.create_publisher(LaserScan, laserscan_topic, 10)

        self.latest_clock_time = None

        self.get_logger().info('PCL Timestamp Sync node started.')

    def clock_callback(self, msg: Clock):
        # 최신 시뮬레이션 시간을 저장 (msg.clock는 builtin_interfaces/Time 타입)
        self.latest_clock_time = msg.clock

    def pcl_callback(self, msg: PointCloud2):
        if self.latest_clock_time is None:
            self.get_logger().warn('아직 /clock 메시지를 수신하지 못했습니다. 업데이트를 건너뜁니다.')
            return

        # 수신된 pcl 메시지를 deepcopy하여 수정
        new_msg = copy.deepcopy(msg)
        # 헤더의 stamp를 /clock에서 받은 최신 시뮬레이션 시간으로 대체
        new_msg.header.stamp = self.latest_clock_time

        # 수정된 메시지를 재발행
        self.publisher_pcl.publish(new_msg)


        # ---------------- LaserScan 변환 시작 ----------------
        # LaserScan 메시지 생성 및 기본 파라미터 설정
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.latest_clock_time
        # LaserScan 메시지의 frame_id는 입력 PointCloud2의 frame_id를 그대로 사용
        scan_msg.header.frame_id = self.sensor_frame

        scan_msg.angle_min = self.get_parameter('scan_angle_min').get_parameter_value().double_value
        scan_msg.angle_max = self.get_parameter('scan_angle_max').get_parameter_value().double_value
        scan_msg.angle_increment = self.get_parameter('scan_angle_increment').get_parameter_value().double_value
        scan_msg.range_min = self.get_parameter('scan_range_min').get_parameter_value().double_value
        scan_msg.range_max = self.get_parameter('scan_range_max').get_parameter_value().double_value

        num_beams = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment) + 1
        # 초기값은 무한대로 설정 (해당 beam에 점이 없으면 range는 inf)
        ranges = [float('inf')] * num_beams

        # 높이 기준 값과 토러스
        base_link_height = self.get_parameter('base_link_height').get_parameter_value().double_value
        rsd455_height = self.get_parameter('rsd455_height').get_parameter_value().double_value
        height_tol = self.get_parameter('height_tolerance').get_parameter_value().double_value

        # PointCloud2 데이터에서 (x, y, z) 좌표 읽기 (nan은 건너뜁니다)
        for point in pc2.read_points(new_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point

            # IsaacSim 기준: 수평면은 xz, 세로(높이)는 y
            # base_link 기준 혹은 RSD455 기준 높이에서 ±height_tol 내에 있는 점들만 선택
            if (abs(y - base_link_height) <= height_tol) or (abs(y - rsd455_height) <= height_tol):
                # 2D 평면상에서의 각도와 거리 계산 (z가 전방, x가 좌우)
                angle = math.atan2(x, z) + PI
                # LaserScan 설정 각 범위 내의 점만 고려
                # if angle < scan_msg.angle_min or angle > scan_msg.angle_max:
                #     continue

                r = math.sqrt(x * x + z * z)
                # 해당 각도의 beam 인덱스 계산
                index = int((angle - scan_msg.angle_min) / scan_msg.angle_increment)
                # 인덱스 안전 검사
                if index < 0 or index >= num_beams:
                    continue
                # 해당 beam에 이미 저장된 거리보다 더 가까운 점이면 업데이트
                if r < ranges[index]:
                    ranges[index] = r

        scan_msg.ranges = ranges

        # LaserScan 메시지 발행
        self.publisher_laserscan.publish(scan_msg)
        # ---------------- LaserScan 변환 끝 ----------------

def main(args=None):
    rclpy.init(args=args)
    node = PCLTimestampSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, 노드를 종료합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
