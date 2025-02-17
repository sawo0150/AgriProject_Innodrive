#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import LaserScan
import tf2_ros
from rosgraph_msgs.msg import Clock  # /clock 메시지 타입
import os
import numpy as np
import transforms3d  # pip install tf-transformations (ROS2에서도 사용 가능)
from scipy.spatial.transform import Rotation as R  # 쿼터니언→회전행렬 변환용
from scipy.spatial import KDTree
import yaml
from PIL import Image

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

def create_2d_transform(u, v, yaw):
    """
    2D 평면상의 변환(회전+이동)을 3x3 호모지니어스 행렬로 생성
    """
    T = np.eye(3)
    T[0, 0] = np.cos(yaw)
    T[0, 1] = -np.sin(yaw)
    T[1, 0] = np.sin(yaw)
    T[1, 1] = np.cos(yaw)
    T[0, 2] = u
    T[1, 2] = v
    return T

def icp_2d(source_points, target_points, init_transform, kdtree, max_iterations=40, tolerance=1e-4):
    """
    간단한 2D ICP 구현  
    - source_points: (N,2) numpy array (레이저 스캔 점들; 센서 좌표계)
    - target_points: (M,2) numpy array (지도 상의 점들)
    - init_transform: (3,3) 초기 추정치 (map←robot) 변환행렬
    - kdtree: target_points에 대해 생성된 KDTree  
    반환: 최종 변환행렬 T (3x3) that best aligns source→target.
    """
    T = init_transform.copy()
    src_h = np.hstack((source_points, np.ones((source_points.shape[0], 1))))  # (N,3)
    for i in range(max_iterations):
        # 현재 T를 적용하여 source 점들을 map 좌표계로 변환
        transformed_src = (T @ src_h.T).T[:, :2]  # (N,2)
        distances, indices = kdtree.query(transformed_src)
        target_corr = target_points[indices]
        # centroids 계산
        centroid_src = np.mean(transformed_src, axis=0)
        centroid_tgt = np.mean(target_corr, axis=0)
        # 중심 이동한 좌표
        src_centered = transformed_src - centroid_src
        tgt_centered = target_corr - centroid_tgt
        # SVD로 최적 회전행렬 계산
        W = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(W)
        R_est = Vt.T @ U.T
        # 반사(reflection) 제거
        if np.linalg.det(R_est) < 0:
            Vt[1, :] *= -1
            R_est = Vt.T @ U.T
        t_est = centroid_tgt - R_est @ centroid_src
        # 업데이트 변환행렬 구성
        T_update = np.eye(3)
        T_update[:2, :2] = R_est
        T_update[:2, 2] = t_est
        # 변환 갱신
        T = T_update @ T
        # 수렴 조건: 번진 이동량이 아주 작으면 종료
        if np.linalg.norm(t_est) < tolerance:
            break
    return T


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


        # 지도 파일(map.yaml) 로드 → 지도 상의 점군(장애물 포인트) 추출
        self.declare_parameter('map_yaml_file', '../maps/nav2_map.yaml')
        map_yaml_file = self.get_parameter('map_yaml_file').get_parameter_value().string_value
        map_yaml_file = os.path.join(current_dir, map_yaml_file)
        map_yaml_file = os.path.abspath(map_yaml_file)
        self.map_points = self.load_map_points(map_yaml_file)

        # 초기 추정치 (localization으로부터 업데이트됨, 2D: 3x3 행렬)
        self.initial_guess_2d = None
        # 최신 laser scan 메시지를 저장 (ICP 연산 시 사용)
        self.latest_scan = None

        # /rtabmap/localization_pose 토픽 구독
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/rtabmap/localization_pose',
            self.localization_pose_callback,
            10
        )
        # 레이저 스캔 토픽 구독 (/scan)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/RPLIDAR_S2E/laser_scan',
            self.laser_scan_callback,
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

        self.get_logger().info('ICPMatchingNode (map->odom TF Publisher) has been started.')


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

    def load_map_points(self, yaml_file):
        """
        YAML 파일을 파싱하여 지도 이미지(PGM/PNG)를 로드하고,
        occupied (장애물) 픽셀들을 월드 좌표계의 2D 점들로 변환한다.
        """
        try:
            with open(yaml_file, 'r') as f:
                map_config = yaml.safe_load(f)
            map_image_file = map_config['image']
            # YAML 파일 경로 기준으로 지도 이미지 파일 경로 계산
            base_dir = os.path.dirname(os.path.abspath(yaml_file))
            map_image_path = os.path.join(base_dir, map_image_file)
            image = Image.open(map_image_path).convert('L')
            map_array = np.array(image)
            resolution = map_config['resolution']
            origin = map_config['origin']  # [x, y, theta]
            negate = map_config.get('negate', 0)
            occupied_thresh = map_config.get('occupied_thresh', 0.65)
            # 이미지 픽셀 값 보정 (negate가 1이면 반전)
            if negate:
                occ_data = 255 - map_array
            else:
                occ_data = map_array
            threshold = occupied_thresh * 255
            obstacles = np.where(occ_data < threshold)  # 장애물: 픽셀 값이 낮은 곳
            points = []
            height = map_array.shape[0]
            for row, col in zip(obstacles[0], obstacles[1]):
                # ROS map_server와 같이 이미지의 (0,0)은 왼쪽 위이므로 y 좌표 반전 적용
                x = col * resolution + origin[0]
                y = (height - row) * resolution + origin[1]
                points.append([x, y])
            points = np.array(points)
            self.get_logger().info(f"Loaded map with {points.shape[0]} obstacle points from {yaml_file}.")
            return points
        except Exception as e:
            self.get_logger().error(f"Failed to load map points: {e}")
            raise

    def timer_callback(self):
        if(self.ts is not None):
            stamp = self.sim_time if self.sim_time is not None else self.get_clock().now().to_msg()
            self.ts.header.stamp = stamp
            self.tf_broadcaster.sendTransform(self.ts)

    def laser_scan_callback(self, msg: LaserScan):
        """
        레이저 스캔 메시지가 올 때마다 최신 scan 데이터를 저장.
        (실제 ICP 연산은 localization callback에서 수행)
        """
        self.latest_scan = msg

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
            self.initial_guess_2d = create_2d_transform(u, v, yaw_2d)
            self.get_logger().info(f"Localization updated initial guess: (u,v)=({u:.2f}, {v:.2f}), yaw={yaw_2d:.2f}")

            # (2) 최신 laser scan 데이터가 있는지 확인 후, ICP 연산 수행
            if self.latest_scan is None:
                self.get_logger().warn("No laser scan data available for ICP. Skipping ICP update.")
                return

            # laser scan 데이터를 polar → Cartesian 변환 (센서 좌표계)
            num_points = len(self.latest_scan.ranges)
            angles = np.linspace(self.latest_scan.angle_min, self.latest_scan.angle_max, num_points)
            ranges = np.array(self.latest_scan.ranges)
            valid = (ranges > self.latest_scan.range_min) & (ranges < self.latest_scan.range_max)
            ranges = ranges[valid]
            angles = angles[valid]
            xs = ranges * np.cos(angles)
            ys = ranges * np.sin(angles)
            scan_points = np.vstack((xs, ys)).T  # (N,2)

            # KDTree 생성 (지도 점군)
            tree = KDTree(self.map_points)
            # ICP 수행: 초기 추정치(self.initial_guess_2d)를 기반으로 scan_points 정합
            T_icp = icp_2d(scan_points, self.map_points, self.initial_guess_2d, tree)
            # ICP 결과로 초기 추정치 업데이트
            self.initial_guess_2d = T_icp

            u_icp = self.initial_guess_2d[0, 2]
            v_icp = self.initial_guess_2d[1, 2]
            yaw_icp = np.arctan2(self.initial_guess_2d[1, 0], self.initial_guess_2d[0, 0])
            
            self.get_logger().info("ICP alignment updated the transform: (u,v)=({u_icp:.2f}, {v_icp:.2f}), yaw={yaw_icp:.2f}")

            # tf publish (RSD455 센서가 지면에서 1.085 m 위에 있으므로 z축 보정)
            self.publish_tf(u_icp, v_icp, yaw_icp, msg.header.stamp)
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
