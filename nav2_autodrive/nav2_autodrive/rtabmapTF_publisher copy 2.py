#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import tf2_ros
from rosgraph_msgs.msg import Clock  # /clock 메시지 타입
import os
import numpy as np
import transforms3d  # pip install tf-transformations (ROS2에서도 사용 가능)
from scipy.spatial.transform import Rotation as R  # 쿼터니언→회전행렬 변환용

PUBLSIH_FREQUENCY = 60

# --- 헬퍼 함수들 ---
def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=float)
    q = np.concatenate(([q[3]], q[:3]))
    R = transforms3d.quaternions.quat2mat(q)
    T = np.eye(4)
    T[:3, :3] = R
    return T

def quaternion_from_matrix(matrix):
    R = matrix[:3, :3]
    q = transforms3d.quaternions.mat2quat(R)
    return np.array([q[1], q[2], q[3], q[0]])

def pose_msg_to_matrix(pose_msg):
    """
    geometry_msgs/Pose를 4x4 numpy 행렬로 변환
    """
    t = np.array([pose_msg.position.x,
                  pose_msg.position.y,
                  pose_msg.position.z])
    quat = np.array([pose_msg.orientation.x,
                     pose_msg.orientation.y,
                     pose_msg.orientation.z,
                     pose_msg.orientation.w])
    matrix = quaternion_matrix(quat)
    matrix[0:3, 3] = t
    return matrix


def transform_msg_to_matrix(transform_msg: TransformStamped):
    """
    geometry_msgs/TransformStamped 메시지를 4x4 numpy 행렬로 변환
    """
    t = np.array([transform_msg.transform.translation.x,
                  transform_msg.transform.translation.y,
                  transform_msg.transform.translation.z])
    quat = np.array([transform_msg.transform.rotation.x,
                     transform_msg.transform.rotation.y,
                     transform_msg.transform.rotation.z,
                     transform_msg.transform.rotation.w])
    matrix = quaternion_matrix(quat)
    matrix[0:3, 3] = t
    return matrix

def matrix_to_transform_stamped(T, parent_frame: str, child_frame: str, stamp):
    """
    4x4 numpy 행렬 T를 TransformStamped 메시지로 변환
    """
    t = T[0:3, 3]
    quat = quaternion_from_matrix(T)
    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent_frame
    ts.child_frame_id = child_frame
    ts.transform.translation.x = t[0]
    ts.transform.translation.y = t[1]
    ts.transform.translation.z = t[2]
    ts.transform.rotation.x = quat[0]
    ts.transform.rotation.y = quat[1]
    ts.transform.rotation.z = quat[2]
    ts.transform.rotation.w = quat[3]
    return ts


class MapOdomTFPublisher(Node):
    def __init__(self):
        super().__init__('map_odom_tf_publisher')

        # 파라미터로 변환 정보 파일 경로를 받음 (기본값: frame_transform_info.npz)
        self.declare_parameter('frame_transform_info_file', '../src/plane_transform_info.npz')
        transform_file = self.get_parameter('frame_transform_info_file').get_parameter_value().string_value

        # 실행 중인 파일의 위치를 기반으로 경로 계산
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # print(current_dir)
        transform_file = os.path.join(current_dir, transform_file)
        transform_file = os.path.abspath(transform_file)  # 절대 경로로 변환

        # plane_params와 transformation_matrix 로드
        self.plane_params, self.transformation_matrix = self.load_plane_transform_info(transform_file)

        # /rtabmap/localization_pose 토픽 구독
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/rtabmap/localization_pose',
            self.localization_pose_callback,
            10
        )
        # TF 브로드캐스터 생성
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info('Map->Odom TF Publisher Node has been started.')

        # tf listener용 버퍼 생성 (odom->RSD455 tf lookup)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # /clock 토픽 구독 (시뮬레이션 시간 구독)
        self.sim_time = None
        self.create_subscription(Clock, '/clock', self.clock_callback, 10)
        self.get_logger().info("Subscribed to /clock for sim time")

        # 20Hz로 publish하기 위한 타이머 (0.05초 주기)
        self.publish_timer = self.create_timer(1/PUBLSIH_FREQUENCY, self.timer_callback)
        self.ts = None

    def clock_callback(self, msg: Clock):
        # /clock 메시지의 시뮬레이션 시간을 저장
        self.sim_time = msg.clock

    def load_plane_transform_info(self, file_path):
        """
        npz파일에서 plane_params, transformation_matrix 등을 읽어온다.
        예: plane_transform_info.npz 안에
            - plane_params = [a, b, c]
            - transformation_matrix = 4x4
        """
        try:
            print(file_path)
            data = np.load(file_path)
            plane_params = data['plane_params']  # shape=(3,)
            transformation_matrix = data['transformation_matrix']  # shape=(4,4)
            self.get_logger().info(f"Loaded plane_transform_info from {file_path}.")
            return plane_params, transformation_matrix
        except Exception as e:
            self.get_logger().error(f"Failed to load plane_transform_info.npz: {e}")
            raise

    def timer_callback(self):
        if(self.ts is not None):
            stamp = self.sim_time if self.sim_time is not None else self.get_clock().now().to_msg()
            self.ts.header.stamp = stamp
            self.tf_broadcaster.sendTransform(self.ts)


    def localization_pose_callback(self, msg):
        """ /rtabmap/localization_pose 메시지 수신 시 호출 """
        try:
            # 3D 포즈 정보 추출
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            # 3D 위치를 평면으로 수직투영하여 2D 좌표 (u, v) 계산
            u, v = self.project_position_to_2d(x, y, z)
            # 쿼터니언을 이용해 평면 상에서의 yaw(회전각) 계산
            yaw_2d = self.compute_2d_yaw_in_plane(qx, qy, qz, qw)

            # tf publish (RSD455 센서가 지면에서 1.085 m 위에 있으므로 z축 보정)
            self.publish_tf(u, v, yaw_2d, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"Error in localization_pose_callback: {e}")

    def project_position_to_2d(self, x, y, z):
        """
        (1) 평면 파라미터 [a, b, c]를 사용해 3D 점을 평면 (1 = a*x+b*y+c*z) 위로 수직투영  
        (2) transformation_matrix의 회전행렬(R)을 이용해 world 좌표계 → 평면 좌표계로 변환  
        반환: 평면 좌표계에서의 (u,v)
        """
        a, b, c = self.plane_params
        denom = a*a + b*b + c*c
        t = (1.0 - (a*x + b*y + c*z)) / denom
        px = x + a*t
        py = y + b*t
        pz = z + c*t

        # transformation_matrix 상단 3x3 (world→plane 회전행렬)
        Rmat = self.transformation_matrix[:3, :3]
        local_3d = Rmat.T @ np.array([px, py, pz])
        # local_3d = [u, v, (높이 거의 0)]
        return local_3d[0], local_3d[1]

    def compute_2d_yaw_in_plane(self, qx, qy, qz, qw):
        """
        쿼터니언 → 회전행렬 변환 후, 로컬 좌표계의 전방 벡터 ([0,0,-1])를 월드 전방 벡터로 변환  
        이를 다시 transformation_matrix의 회전행렬(R)을 이용해 평면 좌표계로 변환한 후,  
        2D에서 atan2를 통해 yaw 각도를 계산한다.
        """
        # (1) 쿼터니언으로 월드 전방 벡터 계산 (여기서는 [0, 0, -1]을 전방으로 가정)
        rot = R.from_quat([qx, qy, qz, qw])
        forward_world = rot.apply([0.0, 0.0, -1.0])
        # (2) world → plane 변환
        Rmat = self.transformation_matrix[:3, :3]
        forward_plane_3d = Rmat.T @ forward_world
        fx, fy = forward_plane_3d[0], forward_plane_3d[1]
        yaw = np.arctan2(fy, fx)
        return yaw

    def publish_tf(self, u, v, yaw, stamp):
        """
        계산된 2D (u,v) 및 yaw를 이용하여,  
        RSD455가 지면보다 1.085 m 위에 있다는 점을 반영해 translation.z에 -1.085를 적용하고,  
        "map" → "odom" tf를 publish한다.
        """

        # 3. tf lookup: odom -> RSD455 transform (Isaac Sim에서 publish됨)
        try:
            # 'odom' frame에서 'RSD455' frame까지의 최신 tf를 lookup (최대 0.5초 기다림)
            # (주의: lookup_transform(target, source, time, timeout))
            trans = self.tf_buffer.lookup_transform('odom', 'RSD455',
                                                    rclpy.time.Time(),
                                                    timeout=rclpy.duration.Duration(seconds=0.5))
            T_odom_robot = transform_msg_to_matrix(trans)
        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform from 'odom' to 'RSD455': {e}")
            return


        # 4. map->odom tf 계산
        #    T_map_odom = T_nav2_robot * inv(T_odom_robot)
        try:
            T_odom_robot_inv = np.linalg.inv(T_odom_robot)
        except np.linalg.LinAlgError as e:
            self.get_logger().error(f"Failed to invert T_odom_robot: {e}")
            return
        
        t_map_robot = TransformStamped()
        t_map_robot.header.stamp = stamp
        t_map_robot.header.frame_id = "map"    # Nav2 기준 map frame
        t_map_robot.child_frame_id = "odom"    # odom frame (Isaac Sim에서 publish하는 odom과 연결)

        t_map_robot.transform.translation.x = u
        t_map_robot.transform.translation.y = v
        t_map_robot.transform.translation.z = 1.085  # RSD455가 지면보다 1.085m 높으므로 지면 좌표로 보정

        # 2D yaw를 이용한 단순 회전 (z축 회전)
        qz_val = np.sin(yaw / 2.0)
        qw_val = np.cos(yaw / 2.0)
        t_map_robot.transform.rotation.x = 0.0
        t_map_robot.transform.rotation.y = 0.0
        t_map_robot.transform.rotation.z = qz_val
        t_map_robot.transform.rotation.w = qw_val
        T_map_robot = transform_msg_to_matrix(t_map_robot)

        T_map_odom = T_map_robot @ T_odom_robot_inv

        stamp = self.sim_time if self.sim_time is not None else self.get_clock().now().to_msg()
        self.ts = matrix_to_transform_stamped(T_map_odom, parent_frame='map', child_frame='odom', stamp=stamp)
        self.get_logger().info(
            f"Published map->odom tf: translation=({u:.2f}, {v:.2f}, {-1.085:.2f}), yaw={yaw:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MapOdomTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down MapOdomTFPublisher Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
