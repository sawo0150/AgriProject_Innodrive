#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from rosgraph_msgs.msg import Clock
import copy, math

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
        # self.publisher_laserscan = self.create_publisher(LaserScan, laserscan_topic, 10)

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
